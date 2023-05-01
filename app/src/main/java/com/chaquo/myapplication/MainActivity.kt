package com.chaquo.myapplication

import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.chaquo.python.PyException
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (! Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }
        val py = Python.getInstance()
        val module = py.getModule("plot")

        findViewById<Button>(R.id.button).setOnClickListener {
            try {
                val pythonInput = intArrayOf(1, 2, 3, 4)
                val bytes = module.callAttr("main", pythonInput)
                Log.d("ptj", bytes.toString())
//                    .toJava(ByteArray::class.java)
//                val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
//                findViewById<ImageView>(R.id.imageView).setImageBitmap(bitmap)
//                println(bitmap.toString())
//                currentFocus?.let {
//                    (getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager)
//                        .hideSoftInputFromWindow(it.windowToken, 0)
//                }
            } catch (e: PyException) {
                Toast.makeText(this, e.message, Toast.LENGTH_LONG).show()
            }
        }
    }
}


