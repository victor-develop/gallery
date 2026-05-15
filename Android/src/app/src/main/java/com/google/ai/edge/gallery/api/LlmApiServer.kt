/*
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ai.edge.gallery.api

import android.util.Base64
import android.util.Log
import com.google.ai.edge.gallery.ui.llmchat.LlmModelInstance
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import com.google.ai.edge.litertlm.SamplerConfig
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import com.google.gson.JsonDeserializationContext
import com.google.gson.JsonDeserializer
import com.google.gson.JsonElement
import com.google.gson.annotations.SerializedName
import fi.iki.elonen.NanoHTTPD
import java.io.PipedInputStream
import java.io.PipedOutputStream
import java.lang.reflect.Type
import java.net.Inet4Address
import java.net.NetworkInterface
import java.util.UUID
import java.util.concurrent.CountDownLatch
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

private const val TAG = "LlmApiServer"
const val API_SERVER_PORT = 8080

// ── Request / response data classes ──────────────────────────────────────────

/**
 * A single content part inside a message. In the OpenAI vision format the `content` field is an
 * array of these objects.
 */
data class ContentPart(
  val type: String = "",
  val text: String? = null,
  @SerializedName("image_url") val imageUrl: ImageUrlPart? = null,
)

data class ImageUrlPart(val url: String = "")

/**
 * Represents the `content` field of a chat message which can be either a plain string or an array
 * of [ContentPart] objects (OpenAI vision format).
 */
sealed class MessageContent {
  data class Text(val text: String) : MessageContent()

  data class Parts(val parts: List<ContentPart>) : MessageContent()
}

/** Custom Gson deserializer that handles the polymorphic `content` field. */
class MessageContentDeserializer : JsonDeserializer<MessageContent> {
  override fun deserialize(
    json: JsonElement,
    typeOfT: Type,
    context: JsonDeserializationContext,
  ): MessageContent =
    if (json.isJsonArray) {
      val parts = json.asJsonArray.map { context.deserialize<ContentPart>(it, ContentPart::class.java) }
      MessageContent.Parts(parts)
    } else {
      MessageContent.Text(if (json.isJsonPrimitive) json.asJsonPrimitive.asString else "")
    }
}

data class ApiChatMessage(
  val role: String = "",
  val content: MessageContent = MessageContent.Text(""),
)

data class ChatCompletionRequest(
  val model: String = "",
  val messages: List<ApiChatMessage> = emptyList(),
  val stream: Boolean = false,
  @SerializedName("max_tokens") val maxTokens: Int? = null,
  val temperature: Double? = null,
  @SerializedName("top_p") val topP: Double? = null,
  @SerializedName("top_k") val topK: Int? = null,
)

// ── Singleton manager ─────────────────────────────────────────────────────────

object LlmApiServerManager {
  private var server: LlmApiServer? = null

  private val _serverUrl = MutableStateFlow<String?>(null)
  val serverUrl: StateFlow<String?> = _serverUrl.asStateFlow()

  @OptIn(ExperimentalApi::class)
  fun startWithModel(instance: LlmModelInstance, modelName: String) {
    server?.stop()
    val newServer = LlmApiServer(API_SERVER_PORT)
    newServer.modelInstance = instance
    newServer.modelName = modelName
    try {
      newServer.start(NanoHTTPD.SOCKET_READ_TIMEOUT, false)
      val ip = getLocalIpAddress() ?: "localhost"
      _serverUrl.value = "http://$ip:$API_SERVER_PORT"
      Log.i(TAG, "API server started at ${_serverUrl.value}")
    } catch (e: Exception) {
      Log.e(TAG, "Failed to start API server", e)
      _serverUrl.value = null
    }
    server = newServer
  }

  fun stop() {
    server?.stop()
    server = null
    _serverUrl.value = null
    Log.i(TAG, "API server stopped")
  }

  private fun getLocalIpAddress(): String? =
    try {
      NetworkInterface.getNetworkInterfaces()
        ?.asSequence()
        ?.flatMap { it.inetAddresses.asSequence() }
        ?.firstOrNull { !it.isLoopbackAddress && it is Inet4Address }
        ?.hostAddress
    } catch (e: Exception) {
      null
    }
}

// ── HTTP server ───────────────────────────────────────────────────────────────

@OptIn(ExperimentalApi::class)
class LlmApiServer(port: Int) : NanoHTTPD(port) {
  private val gson: Gson =
    GsonBuilder()
      .registerTypeAdapter(MessageContent::class.java, MessageContentDeserializer())
      .create()
  private val inferencing = AtomicBoolean(false)

  @Volatile var modelInstance: LlmModelInstance? = null
  @Volatile var modelName: String = "local-model"

  override fun serve(session: IHTTPSession): Response {
    val response =
      when {
        session.method == Method.OPTIONS -> newFixedLengthResponse(Response.Status.OK, MIME_PLAINTEXT, "")
        session.method == Method.GET && session.uri == "/v1/models" -> handleModels()
        session.method == Method.POST && session.uri == "/v1/chat/completions" ->
          handleChatCompletions(session)
        else ->
          jsonResponse(
            Response.Status.NOT_FOUND,
            """{"error":{"message":"Not found","type":"invalid_request_error","code":"not_found"}}""",
          )
      }
    response.addHeader("Access-Control-Allow-Origin", "*")
    response.addHeader("Access-Control-Allow-Headers", "Content-Type, Authorization")
    response.addHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    return response
  }

  private fun handleModels(): Response {
    val ts = System.currentTimeMillis() / 1000
    val data =
      if (modelInstance != null)
        """[{"id":"$modelName","object":"model","created":$ts,"owned_by":"local"}]"""
      else "[]"
    return jsonResponse(Response.Status.OK, """{"object":"list","data":$data}""")
  }

  private fun handleChatCompletions(session: IHTTPSession): Response {
    val instance =
      modelInstance
        ?: return jsonResponse(
          Response.Status.SERVICE_UNAVAILABLE,
          """{"error":{"message":"No model is loaded","type":"server_error"}}""",
        )

    val contentLength = session.headers["content-length"]?.toIntOrNull() ?: 0
    if (contentLength == 0) {
      return jsonResponse(
        Response.Status.BAD_REQUEST,
        """{"error":{"message":"Empty request body","type":"invalid_request_error"}}""",
      )
    }
    val bodyBytes = ByteArray(contentLength)
    session.inputStream.read(bodyBytes, 0, contentLength)

    val request =
      try {
        gson.fromJson(String(bodyBytes), ChatCompletionRequest::class.java)
          ?: return jsonResponse(
            Response.Status.BAD_REQUEST,
            """{"error":{"message":"Invalid JSON","type":"invalid_request_error"}}""",
          )
      } catch (e: Exception) {
        return jsonResponse(
          Response.Status.BAD_REQUEST,
          """{"error":{"message":"JSON parse error: ${e.message?.escapeJson()}","type":"invalid_request_error"}}""",
        )
      }

    if (request.messages.isEmpty()) {
      return jsonResponse(
        Response.Status.BAD_REQUEST,
        """{"error":{"message":"messages cannot be empty","type":"invalid_request_error"}}""",
      )
    }

    if (!inferencing.compareAndSet(false, true)) {
      return jsonResponse(
        Response.Status.TOO_MANY_REQUESTS,
        """{"error":{"message":"A request is already in progress. Please try again later.","type":"server_error"}}""",
      )
    }

    return if (request.stream) {
      handleStreaming(instance, request) // thread resets inferencing
    } else {
      try {
        handleNonStreaming(instance, request)
      } finally {
        inferencing.set(false)
      }
    }
  }

  private fun handleNonStreaming(
    instance: LlmModelInstance,
    request: ChatCompletionRequest,
  ): Response {
    val parsed = extractMessages(request.messages)

    val conversation =
      try {
        createConversation(instance, parsed.systemPrompt, request)
      } catch (e: Exception) {
        return jsonResponse(
          Response.Status.INTERNAL_ERROR,
          """{"error":{"message":"Failed to create conversation: ${e.message?.escapeJson()}","type":"server_error"}}""",
        )
      }

    val text = StringBuilder()
    val latch = CountDownLatch(1)
    val errorRef = AtomicReference<String?>(null)

    conversation.sendMessageAsync(
      buildContents(parsed.userPrompt, parsed.imageBytes),
      object : MessageCallback {
        override fun onMessage(message: Message) {
          val token = message.toString()
          if (!token.startsWith("<ctrl")) text.append(token)
        }

        override fun onDone() = latch.countDown()

        override fun onError(throwable: Throwable) {
          errorRef.set(throwable.message ?: "Inference error")
          latch.countDown()
        }
      },
      emptyMap(),
    )

    latch.await()
    try {
      conversation.close()
    } catch (_: Exception) {}

    errorRef.get()?.let { err ->
      return jsonResponse(
        Response.Status.INTERNAL_ERROR,
        """{"error":{"message":"${err.escapeJson()}","type":"server_error"}}""",
      )
    }

    val id = "chatcmpl-${UUID.randomUUID()}"
    val ts = System.currentTimeMillis() / 1000
    val contentJson = gson.toJson(text.toString())
    return jsonResponse(
      Response.Status.OK,
      """{"id":"$id","object":"chat.completion","created":$ts,"model":"$modelName","choices":[{"index":0,"message":{"role":"assistant","content":$contentJson},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}""",
    )
  }

  private fun handleStreaming(instance: LlmModelInstance, request: ChatCompletionRequest): Response {
    val parsed = extractMessages(request.messages)
    val id = "chatcmpl-${UUID.randomUUID()}"
    val ts = System.currentTimeMillis() / 1000

    val pipedIn = PipedInputStream(65536)
    val pipedOut = PipedOutputStream(pipedIn)

    Thread {
      try {
        val conversation = createConversation(instance, parsed.systemPrompt, request)
        val latch = CountDownLatch(1)

        // Send the role delta first (OpenAI spec)
        val roleDelta =
          """{"id":"$id","object":"chat.completion.chunk","created":$ts,"model":"$modelName","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}"""
        pipedOut.write("data: $roleDelta\n\n".toByteArray())
        pipedOut.flush()

        conversation.sendMessageAsync(
          buildContents(parsed.userPrompt, parsed.imageBytes),
          object : MessageCallback {
            override fun onMessage(message: Message) {
              val token = message.toString()
              if (!token.startsWith("<ctrl")) {
                val contentJson = gson.toJson(token)
                // Remove surrounding quotes from gson output
                val inner = contentJson.substring(1, contentJson.length - 1)
                val chunk =
                  """{"id":"$id","object":"chat.completion.chunk","created":$ts,"model":"$modelName","choices":[{"index":0,"delta":{"content":"$inner"},"finish_reason":null}]}"""
                pipedOut.write("data: $chunk\n\n".toByteArray())
                pipedOut.flush()
              }
            }

            override fun onDone() {
              val stopChunk =
                """{"id":"$id","object":"chat.completion.chunk","created":$ts,"model":"$modelName","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"""
              pipedOut.write("data: $stopChunk\n\ndata: [DONE]\n\n".toByteArray())
              pipedOut.flush()
              latch.countDown()
            }

            override fun onError(throwable: Throwable) {
              Log.e(TAG, "Streaming inference error", throwable)
              pipedOut.write("data: [DONE]\n\n".toByteArray())
              pipedOut.flush()
              latch.countDown()
            }
          },
          emptyMap(),
        )

        latch.await()
        try {
          conversation.close()
        } catch (_: Exception) {}
      } catch (e: Exception) {
        Log.e(TAG, "Streaming thread error", e)
      } finally {
        try {
          pipedOut.close()
        } catch (_: Exception) {}
        inferencing.set(false)
      }
    }.also { it.isDaemon = true }.start()

    return newChunkedResponse(Response.Status.OK, "text/event-stream", pipedIn)
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  /** Parsed result of [extractMessages]. */
  private data class ParsedMessages(
    val systemPrompt: String?,
    val userPrompt: String,
    val imageBytes: List<ByteArray>,
  )

  /**
   * Extracts a system prompt, the user prompt text, and any base64-encoded images from the OpenAI
   * message list. Images supplied via `image_url` data-URIs are decoded and returned as raw bytes.
   */
  private fun extractMessages(messages: List<ApiChatMessage>): ParsedMessages {
    val system =
      messages
        .firstOrNull { it.role == "system" }
        ?.let { messageText(it.content) }
        ?.takeIf { it.isNotEmpty() }

    val others = messages.filter { it.role != "system" }
    val images = mutableListOf<ByteArray>()

    // Collect images from all user messages.
    for (msg in others) {
      when (val c = msg.content) {
        is MessageContent.Parts ->
          c.parts
            .filter { it.type == "image_url" }
            .mapNotNull { it.imageUrl?.url }
            .mapNotNull { decodeDataUri(it) }
            .forEach { images.add(it) }
        is MessageContent.Text -> {} // no images in plain text
      }
    }

    val prompt =
      if (others.size == 1) {
        messageText(others.first().content)
      } else {
        // Multi-turn: format as labelled dialogue.
        buildString {
          others.forEachIndexed { i, msg ->
            val label = if (msg.role == "assistant") "Assistant" else "User"
            if (i > 0) append("\n\n")
            append("$label: ${messageText(msg.content)}")
          }
        }
      }

    return ParsedMessages(systemPrompt = system, userPrompt = prompt, imageBytes = images)
  }

  /** Extracts the text from a [MessageContent], concatenating text parts if needed. */
  private fun messageText(content: MessageContent): String =
    when (content) {
      is MessageContent.Text -> content.text
      is MessageContent.Parts ->
        content.parts.filter { it.type == "text" }.mapNotNull { it.text }.joinToString("\n")
    }

  /**
   * Decodes a `data:image/...;base64,...` URI to raw bytes. Returns `null` for non-data URIs or on
   * decode failure.
   */
  private fun decodeDataUri(url: String): ByteArray? {
    if (!url.startsWith("data:")) return null
    val commaIdx = url.indexOf(',')
    if (commaIdx < 0) return null
    return try {
      Base64.decode(url.substring(commaIdx + 1), Base64.DEFAULT)
    } catch (e: Exception) {
      Log.w(TAG, "Failed to decode base64 image data", e)
      null
    }
  }

  /** Builds the [Contents] to send to the model, placing images before text. */
  private fun buildContents(userPrompt: String, imageBytes: List<ByteArray>): Contents {
    val parts = mutableListOf<Content>()
    for (img in imageBytes) {
      parts.add(Content.ImageBytes(img))
    }
    if (userPrompt.isNotBlank()) {
      parts.add(Content.Text(userPrompt))
    }
    // Ensure there is always at least one content part for the model.
    if (parts.isEmpty()) {
      parts.add(Content.Text(""))
    }
    return Contents.of(parts)
  }

  private fun createConversation(
    instance: LlmModelInstance,
    systemPrompt: String?,
    request: ChatCompletionRequest,
  ) =
    instance.engine.createConversation(
      ConversationConfig(
        samplerConfig =
          SamplerConfig(
            topK = request.topK ?: 40,
            topP = request.topP ?: 0.95,
            temperature = request.temperature ?: 0.8,
          ),
        systemInstruction =
          if (!systemPrompt.isNullOrEmpty()) Contents.of(systemPrompt) else null,
      )
    )

  private fun jsonResponse(status: Response.Status, body: String): Response =
    newFixedLengthResponse(status, "application/json", body)

  private fun String.escapeJson(): String =
    this.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "\\r")
}
