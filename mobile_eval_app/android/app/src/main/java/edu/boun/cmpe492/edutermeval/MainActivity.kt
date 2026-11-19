package edu.boun.cmpe492.edutermeval

import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.webkit.JavascriptInterface
import android.webkit.WebChromeClient
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.appcompat.app.AppCompatActivity
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {

    private lateinit var webView: WebView

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        webView = WebView(this)
        setContentView(webView)

        val settings: WebSettings = webView.settings
        settings.javaScriptEnabled = true
        settings.domStorageEnabled = true
        settings.allowFileAccess = true

        webView.webViewClient = WebViewClient()
        webView.webChromeClient = WebChromeClient()
        webView.addJavascriptInterface(AndroidBridge(this), "AndroidBridge")
        webView.loadUrl("file:///android_asset/index.html")
    }

    override fun onBackPressed() {
        if (webView.canGoBack()) {
            webView.goBack()
        } else {
            super.onBackPressed()
        }
    }

    inner class AndroidBridge(private val ctx: Context) {
        private val prefs = ctx.getSharedPreferences("eduterm_eval_prefs", Context.MODE_PRIVATE)

        @JavascriptInterface
        fun saveDraft(jsonStr: String) {
            prefs.edit().putString("draft", jsonStr).apply()
        }

        @JavascriptInterface
        fun loadDraft(): String {
            return prefs.getString("draft", "") ?: ""
        }

        @JavascriptInterface
        fun shareReport(summaryText: String, jsonStr: String) {
            val payload = "$summaryText\n\n--- HAM TEST VERISI (JSON) ---\n$jsonStr"
            val intent = Intent(Intent.ACTION_SEND).apply {
                type = "text/plain"
                putExtra(Intent.EXTRA_SUBJECT, "CMPE 492 EduTerm Öğrenci Test Verisi")
                putExtra(Intent.EXTRA_TEXT, payload)
            }
            ctx.startActivity(Intent.createChooser(intent, "Test Verisini Gönder / Paylaş"))
        }

        @JavascriptInterface
        fun postJson(serverUrl: String, jsonStr: String, callbackFn: String) {
            thread {
                var ok = false
                var msg = ""
                try {
                    val cleanUrl = serverUrl.trim().trimEnd('/') + "/submit"
                    val conn = (URL(cleanUrl).openConnection() as HttpURLConnection).apply {
                        requestMethod = "POST"
                        connectTimeout = 8000
                        readTimeout = 8000
                        doOutput = true
                        setRequestProperty("Content-Type", "application/json; charset=UTF-8")
                    }
                    OutputStreamWriter(conn.outputStream, Charsets.UTF_8).use { it.write(jsonStr) }
                    val code = conn.responseCode
                    val body = (if (code in 200..299) conn.inputStream else conn.errorStream)
                        ?.bufferedReader()?.readText() ?: ""
                    ok = code in 200..299
                    msg = if (ok) "Sunucuya kaydedildi (HTTP $code)" else "Hata HTTP $code: $body"
                    conn.disconnect()
                } catch (e: Exception) {
                    msg = "Bağlantı hatası: ${e.localizedMessage ?: e.javaClass.simpleName}"
                }
                val escaped = msg.replace("\\", "\\\\").replace("'", "\\'")
                runOnUiThread {
                    webView.evaluateJavascript("$callbackFn($ok, '$escaped')", null)
                }
            }
        }
    }
}
