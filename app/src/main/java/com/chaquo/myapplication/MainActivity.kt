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
        // binds to the activity_main layout file (currently not really used for this demo)
        setContentView(R.layout.activity_main)

        if (! Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }
        val py = Python.getInstance()
        // runs the python file called plot.py in the python folder
        val module = py.getModule("plot")

        // this is just a test app
        // the result from the models are all being outputted in the logcat (nothing is being displayed
        // or used on the app gui)
        // NOTE: inside the python file, you can call short_data_25.csv (~1 min execution) or short_short_data_25.csv
        // (~6 min execution)
        // FUTURE WORK: currently changesToGraph is being run using a ptl file but GCNModelVAE is being
        // run directly with torch on the app. It may execute faster if that is turned into a ptl file
    }
}


