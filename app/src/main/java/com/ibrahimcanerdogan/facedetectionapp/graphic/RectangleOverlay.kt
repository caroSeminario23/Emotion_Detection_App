package com.ibrahimcanerdogan.facedetectionapp.graphic

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import com.google.mlkit.vision.face.Face
import com.ibrahimcanerdogan.facedetectionapp.utils.CameraUtils

class RectangleOverlay(
    private val overlay: GraphicOverlay<*>,
    private val face : Face,
    private val rect : Rect
) : GraphicOverlay.Graphic(overlay) {

    private val boxPaint : Paint = Paint()
    private val textPaint: Paint = Paint()
    private var text: String = ""

    init {
        boxPaint.color = Color.GREEN
        boxPaint.style = Paint.Style.STROKE
        boxPaint.strokeWidth = 3.0f

        textPaint.color = Color.GREEN
        textPaint.textSize = 40.0f
        textPaint.style = Paint.Style.FILL
    }

    override fun draw(canvas: Canvas) {
        val rect = CameraUtils.calculateRect(
            overlay,
            rect.height().toFloat(),
            rect.width().toFloat(),
            face.boundingBox
        )
        canvas.drawRect(rect, boxPaint)
        if (text.isNotEmpty()) {
            canvas.drawText(text, rect.left.toFloat(), rect.top.toFloat() - 10, textPaint)
        }
    }

    fun setText(text: String) {
        this.text = text
    }

    fun getBitmap(): Bitmap {
        val rect = CameraUtils.calculateRect(
            overlay,
            rect.height().toFloat(),
            rect.width().toFloat(),
            face.boundingBox
        )
        val bitmap = Bitmap.createBitmap(rect.width().toInt(), rect.height().toInt(), Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        draw(canvas)
        return bitmap
    }
}