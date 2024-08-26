package com.example.ex

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Button
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.example.ex.ui.theme.ExTheme
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream

class MainActivity : ComponentActivity() {

    private lateinit var model: Module

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        // loading model
        model = Module.load(assetFilePath("semivl.pt"))

        setContent {
            ExTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    MainContent(
                        modifier = Modifier.padding(innerPadding),
                        model = model
                    )
                }
            }
        }
    }

    private fun assetFilePath(assetName: String): String {
        val file = File(filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }
        assets.open(assetName).use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
                outputStream.flush()
            }
        }
        return file.absolutePath
    }
}

@Composable
fun MainContent(modifier: Modifier = Modifier, model: Module? = null) {
    var originalBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var resultBitmap by remember { mutableStateOf<Bitmap?>(null) }

    val context = LocalContext.current

    Column(
        modifier = modifier.fillMaxSize().padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        originalBitmap?.let {
            Image(
                bitmap = it.asImageBitmap(),
                contentDescription = "Original Image",
                modifier = Modifier.size(300.dp)
            )
        }

        Button(onClick = {
            if (model != null) {
                originalBitmap = BitmapFactory.decodeResource(context.resources, R.drawable.test)
                originalBitmap = originalBitmap?.let { resizeBitmap(it, 300, 300) }
                originalBitmap?.let { bitmap ->
                    resultBitmap = runSegmentation(bitmap, model)
                }
            }
        }) {
            Text(text = "Run Segmentation")
        }

        resultBitmap?.let {
            Image(
                bitmap = it.asImageBitmap(),
                contentDescription = "Segmentation Result",
                modifier = Modifier.size(300.dp)
            )
        }
    }
}

@Preview(showBackground = true)
@Composable
fun MainContentPreview() {
    ExTheme {
        MainContent()
    }
}
private fun runSegmentation(bitmap: Bitmap, model: Module): Bitmap {
    // Preprocess the image
    val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
        bitmap,
        TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
        TensorImageUtils.TORCHVISION_NORM_STD_RGB
    )

    // Run inference
    val outputTensor = model.forward(IValue.from(inputTensor)).toTensor()

    // Log the output tensor shape and data
    Log.d("SegmentationDebug", "Output Tensor Shape: ${outputTensor.shape().contentToString()}")

    // Extract data and shape
    val outputArray = outputTensor.dataAsFloatArray
    val numClasses = outputTensor.shape()[1].toInt()
    val height = outputTensor.shape()[2].toInt()
    val width = outputTensor.shape()[3].toInt()

    // Create a bitmap for the segmentation mask
    val segmentedBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

    // Iterate over each pixel
    for (y in 0 until height) {
        for (x in 0 until width) {
            // Calculate the class scores for this pixel
            var maxIndex = 0
            var maxScore = Float.NEGATIVE_INFINITY
            for (c in 0 until numClasses) {
                val score = outputArray[c * width * height + y * width + x]
                if (score > maxScore) {
                    maxScore = score
                    maxIndex = c
                }
            }

            // Get the color for the class and set the pixel in the bitmap
            val color = getColorForClass(maxIndex)
            segmentedBitmap.setPixel(x, y, color)
        }
    }

    // Resize to match input bitmap size if needed
    return Bitmap.createScaledBitmap(segmentedBitmap, bitmap.width, bitmap.height, true)
}


private fun getColorForClass(classIndex: Int): Int {
    val colors = intArrayOf(
        0x80808080.toInt(), // Gray
        0x80000000.toInt(), // Black
        0x80FF0000.toInt(), // Red
        0x8000FF00.toInt(), // Green
        0x800000FF.toInt(), // Blue
        0x80FFFF00.toInt(), // Yellow
        0x80FF00FF.toInt(), // Magenta
        0x8000FFFF.toInt(), // Cyan
        0x80FF8000.toInt(), // Orange
        0x80008000.toInt(), // Olive
        0x80000080.toInt(), // Purple
        0x80800080.toInt(), // Maroon
        0x80FF8080.toInt(), // Light Red
        0x8080FF80.toInt(), // Light Green
        0x808080FF.toInt(), // Light Blue
        0x80FF0080.toInt(), // Pink
        0x80FFFFFF.toInt(), // White
        0x80808000.toInt(), // Dark Yellow
        0x80008080.toInt(), // Dark Magenta
        0x808000FF.toInt(), // Light Purple
        0x80FF80FF.toInt()  // Light Magenta
    )
    return colors[classIndex % colors.size]
}

private fun resizeBitmap(bitmap: Bitmap, maxWidth: Int, maxHeight: Int): Bitmap {
    val ratioBitmap = bitmap.width.toFloat() / bitmap.height.toFloat()
    val ratioMax = maxWidth.toFloat() / maxHeight.toFloat()

    var finalWidth = maxWidth
    var finalHeight = maxHeight

    if (ratioMax > 1) {
        finalWidth = (maxHeight * ratioBitmap).toInt()
    } else {
        finalHeight = (maxWidth / ratioBitmap).toInt()
    }

    return Bitmap.createScaledBitmap(bitmap, finalWidth, finalHeight, true)
}
