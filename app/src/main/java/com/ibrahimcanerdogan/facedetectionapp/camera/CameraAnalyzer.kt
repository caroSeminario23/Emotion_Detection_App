package com.ibrahimcanerdogan.facedetectionapp.camera

import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.ibrahimcanerdogan.facedetectionapp.graphic.GraphicOverlay
import com.ibrahimcanerdogan.facedetectionapp.graphic.RectangleOverlay
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.io.File
import java.io.FileWriter
import java.io.IOException

class CameraAnalyzer(
    private val overlay: GraphicOverlay<*>,
    modelPath: String
) : BaseCameraAnalyzer<List<Face>>(){

    override val graphicOverlay: GraphicOverlay<*>
        get() = overlay

    private val cameraOptions = FaceDetectorOptions.Builder()
        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
        .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
        .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
        .setMinFaceSize(0.15f)
        .enableTracking()
        .build()

    private val detector = FaceDetection.getClient(cameraOptions)
    private val interpreter: Interpreter

    init {
        // Cargar el modelo TFLite
        val assetFileDescriptor = overlay.context.assets.openFd(modelPath)
        val fileInputStream = assetFileDescriptor.createInputStream()
        val fileChannel = fileInputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        val modelBuffer = fileChannel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        interpreter = Interpreter(modelBuffer)
    }

    override fun detectInImage(image: InputImage): Task<List<Face>> {
        return detector.process(image)
    }

    override fun stop() {
       try {
           detector.close()
           interpreter.close()
       } catch (e : Exception) {
           Log.e(TAG , "stop : $e")
       }
    }

    override fun onSuccess(results: List<Face>, graphicOverlay: GraphicOverlay<*>, rect: Rect) {
        graphicOverlay.clear()
        results.forEach { face ->
            val faceGraphic = RectangleOverlay(graphicOverlay, face, rect)
            graphicOverlay.add(faceGraphic)

            // Preprocesar el rostro
            val faceBitmap = faceGraphic.getBitmap()
            val resizedBitmap = Bitmap.createScaledBitmap(faceBitmap, 48, 48, true)
            val byteBuffer = convertBitmapToByteBuffer(resizedBitmap)

            // Realizar la inferencia
            val output = Array(1) { FloatArray(1) }
            interpreter.run(byteBuffer, output)

            // Procesar la salida
            val estado = if (output[0][0] > 0.5) "Inestable" else "Estable"

            // Mostrar el estado sobre el recuadro
            faceGraphic.setText(estado)

            Log.d(TAG, "Resultado de la predicción: $estado")

            // Guardar el estado en un archivo csv
            saveResultToCSV(estado, output[0][0])
        }
        graphicOverlay.postInvalidate()
    }

    private fun saveResultToCSV(estado: String, output: Float) {
        val file = File(overlay.context.filesDir, "resultados.csv")
        try {
            val writer = FileWriter(file, true)
            writer.append("$estado,$output\n")
            writer.flush()
            writer.close()
        } catch (e: IOException) {
            Log.e(TAG, "Error writing to CSV file: $e")
        }
    }

    override fun onFailure(e: Exception) {
        Log.e(TAG, "onFailure : $e")
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(4 * 48 * 48 * 3)
        byteBuffer.order(ByteOrder.nativeOrder())
        val intValues = IntArray(48 * 48)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        var pixel = 0
        for (i in 0 until 48) {
            for (j in 0 until 48) {
                val value = intValues[pixel++]
                byteBuffer.putFloat(((value shr 16 and 0xFF) / 255.0f))
                byteBuffer.putFloat(((value shr 8 and 0xFF) / 255.0f))
                byteBuffer.putFloat(((value and 0xFF) / 255.0f))
            }
        }
        return byteBuffer
    }

    companion object {
        private const val TAG = "CameraAnalyzer"
    }
}