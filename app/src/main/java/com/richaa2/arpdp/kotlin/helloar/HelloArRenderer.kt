/*
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.richaa2.arpdp.kotlin.helloar

import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import android.opengl.GLES30
import android.opengl.Matrix
import android.util.Log
import androidx.core.graphics.createBitmap
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import com.google.ar.core.Anchor
import com.google.ar.core.Camera
import com.google.ar.core.Frame
import com.google.ar.core.HitResult
import com.google.ar.core.LightEstimate
import com.google.ar.core.Plane
import com.google.ar.core.Pose
import com.google.ar.core.Session
import com.google.ar.core.Trackable
import com.google.ar.core.TrackingFailureReason
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.exceptions.DeadlineExceededException
import com.google.ar.core.exceptions.NotYetAvailableException
import com.richaa2.arpdp.R
import com.richaa2.arpdp.java.common.helpers.DisplayRotationHelper
import com.richaa2.arpdp.java.common.helpers.TrackingStateHelper
import com.richaa2.arpdp.java.common.samplerender.Framebuffer
import com.richaa2.arpdp.java.common.samplerender.GLError
import com.richaa2.arpdp.java.common.samplerender.IndexBuffer
import com.richaa2.arpdp.java.common.samplerender.Mesh
import com.richaa2.arpdp.java.common.samplerender.SampleRender
import com.richaa2.arpdp.java.common.samplerender.Shader
import com.richaa2.arpdp.java.common.samplerender.Texture
import com.richaa2.arpdp.java.common.samplerender.VertexBuffer
import com.richaa2.arpdp.java.common.samplerender.arcore.BackgroundRenderer
import com.richaa2.arpdp.java.common.samplerender.arcore.PlaneRenderer
import com.richaa2.arpdp.java.common.samplerender.arcore.SpecularCubemapFilter
import com.richaa2.arpdp.kotlin.common.MeasurementMode
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.IntBuffer
import kotlin.math.pow
import kotlin.math.sqrt

/** Renders the HelloAR application using our example Renderer. */
class HelloArRenderer(val activity: HelloArActivity) :
    SampleRender.Renderer, DefaultLifecycleObserver {

    var selectedMode: MeasurementMode = MeasurementMode.TwoPoints

    // Stored view dimensions for center hit-test
    private var viewWidth: Int = 0
    private var viewHeight: Int = 0

    companion object {
        val TAG = "HelloArRenderer"

        // See the definition of updateSphericalHarmonicsCoefficients for an explanation of these
        // constants.
        private val sphericalHarmonicFactors =
            floatArrayOf(
                0.282095f,
                -0.325735f,
                0.325735f,
                -0.325735f,
                0.273137f,
                -0.273137f,
                0.078848f,
                -0.273137f,
                0.136569f
            )

        private val Z_NEAR = 0.1f
        private val Z_FAR = 100f

        // Assumed distance from the device camera to the surface on which user will try to place
        // objects.
        // This value affects the apparent scale of objects while the tracking method of the
        // Instant Placement point is SCREENSPACE_WITH_APPROXIMATE_DISTANCE.
        // Values in the [0.2, 2.0] meter range are a good choice for most AR experiences. Use lower
        // values for AR experiences where users are expected to place objects on surfaces close to the
        // camera. Use larger values for experiences where the user will likely be standing and trying
        // to
        // place an object on the ground or floor in front of them.
        val APPROXIMATE_DISTANCE_METERS = 2.0f

        val CUBEMAP_RESOLUTION = 16
        val CUBEMAP_NUMBER_OF_IMPORTANCE_SAMPLES = 32
    }

    lateinit var render: SampleRender
    lateinit var planeRenderer: PlaneRenderer
    lateinit var backgroundRenderer: BackgroundRenderer
    lateinit var virtualSceneFramebuffer: Framebuffer
    var hasSetTextureNames = false

    // Point Cloud
    lateinit var pointCloudVertexBuffer: VertexBuffer
    lateinit var pointCloudMesh: Mesh
    lateinit var pointCloudShader: Shader

    // Anchor visualization (distance label and plane indicator)
    private lateinit var distanceShader: Shader
    private lateinit var distanceQuadMesh: Mesh
    private lateinit var innerCircleShader: Shader
    private lateinit var innerCircleMesh: Mesh
    private lateinit var outerCircleShader: Shader
    private lateinit var outerCircleMesh: Mesh
    private var distanceTexture: Texture? = null

    private lateinit var circleFillMesh: Mesh
    private lateinit var circleFillShader: Shader

    private lateinit var dotFillMesh: Mesh

    // --- Line between two anchors ---
    private lateinit var lineShader: Shader
    private lateinit var lineMesh: Mesh
    private lateinit var lineVertexBuffer: VertexBuffer
    private lateinit var lineFloatBuffer: FloatBuffer
    private val vpMatrix = FloatArray(16) // P * V for line rendering (model = identity)

    // Keep track of the last point cloud rendered to avoid updating the VBO if point cloud
    // was not changed.  Do this using the timestamp since we can't compare PointCloud objects.
    var lastPointCloudTimestamp: Long = 0

    // Virtual object (ARCore pawn)
    lateinit var virtualObjectMesh: Mesh
    lateinit var virtualObjectShader: Shader
    lateinit var virtualObjectAlbedoTexture: Texture
    lateinit var virtualObjectAlbedoInstantPlacementTexture: Texture

    private val wrappedAnchors = mutableListOf<WrappedAnchor>()

    // Environmental HDR
    lateinit var dfgTexture: Texture
    lateinit var cubemapFilter: SpecularCubemapFilter

    // Temporary matrix allocated here to reduce number of allocations for each frame.
    val modelMatrix = FloatArray(16)
    val viewMatrix = FloatArray(16)
    val projectionMatrix = FloatArray(16)
    val modelViewMatrix = FloatArray(16) // view x model

    val modelViewProjectionMatrix = FloatArray(16) // projection x view x model

    val sphericalHarmonicsCoefficients = FloatArray(9 * 3)
    val viewInverseMatrix = FloatArray(16)
    val worldLightDirection = floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f)
    val viewLightDirection = FloatArray(4) // view x world light direction

    val session
        get() = activity.arCoreSessionHelper.session

    val displayRotationHelper = DisplayRotationHelper(activity)
    val trackingStateHelper = TrackingStateHelper(activity)

    override fun onResume(owner: LifecycleOwner) {
        displayRotationHelper.onResume()
        hasSetTextureNames = false
    }

    override fun onPause(owner: LifecycleOwner) {
        displayRotationHelper.onPause()
    }

    override fun onSurfaceCreated(render: SampleRender) {
        this.render = render
        // Prepare the rendering objects.
        // This involves reading shaders and 3D model files, so may throw an IOException.
        try {
            planeRenderer = PlaneRenderer(render)
            backgroundRenderer = BackgroundRenderer(render)
            virtualSceneFramebuffer = Framebuffer(render, /*width=*/ 1, /*height=*/ 1)

            cubemapFilter =
                SpecularCubemapFilter(
                    render,
                    CUBEMAP_RESOLUTION,
                    CUBEMAP_NUMBER_OF_IMPORTANCE_SAMPLES
                )
            // Load environmental lighting values lookup table
            dfgTexture =
                Texture(
                    render,
                    Texture.Target.TEXTURE_2D,
                    Texture.WrapMode.CLAMP_TO_EDGE,
                    /*useMipmaps=*/ false
                )
            // The dfg.raw file is a raw half-float texture with two channels.
            val dfgResolution = 64
            val dfgChannels = 2
            val halfFloatSize = 2

            val buffer: ByteBuffer =
                ByteBuffer.allocateDirect(dfgResolution * dfgResolution * dfgChannels * halfFloatSize)
            activity.assets.open("models/dfg.raw").use { it.read(buffer.array()) }

            // SampleRender abstraction leaks here.
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, dfgTexture.textureId)
            GLError.maybeThrowGLException("Failed to bind DFG texture", "glBindTexture")
            GLES30.glTexImage2D(
                GLES30.GL_TEXTURE_2D,
                /*level=*/ 0,
                GLES30.GL_RG16F,
                /*width=*/ dfgResolution,
                /*height=*/ dfgResolution,
                /*border=*/ 0,
                GLES30.GL_RG,
                GLES30.GL_HALF_FLOAT,
                buffer
            )
            GLError.maybeThrowGLException("Failed to populate DFG texture", "glTexImage2D")

            // Point cloud
            pointCloudShader =
                Shader.createFromAssets(
                    render,
                    "shaders/point_cloud.vert",
                    "shaders/point_cloud.frag",
                    /*defines=*/ null
                )
                    .setVec4(
                        "u_Color",
                        floatArrayOf(31.0f / 255.0f, 188.0f / 255.0f, 210.0f / 255.0f, 1.0f)
                    )
                    .setFloat("u_PointSize", 5.0f)

            // four entries per vertex: X, Y, Z, confidence
            pointCloudVertexBuffer =
                VertexBuffer(render, /*numberOfEntriesPerVertex=*/ 4, /*entries=*/ null)
            val pointCloudVertexBuffers = arrayOf(pointCloudVertexBuffer)
            pointCloudMesh =
                Mesh(
                    render,
                    Mesh.PrimitiveMode.POINTS, /*indexBuffer=*/
                    null,
                    pointCloudVertexBuffers
                )

            // Virtual object to render (ARCore pawn)
            virtualObjectAlbedoTexture =
                Texture.createFromAsset(
                    render,
                    "models/pawn_albedo.png",
                    Texture.WrapMode.CLAMP_TO_EDGE,
                    Texture.ColorFormat.SRGB
                )

            virtualObjectAlbedoInstantPlacementTexture =
                Texture.createFromAsset(
                    render,
                    "models/pawn_albedo_instant_placement.png",
                    Texture.WrapMode.CLAMP_TO_EDGE,
                    Texture.ColorFormat.SRGB
                )

            val virtualObjectPbrTexture =
                Texture.createFromAsset(
                    render,
                    "models/pawn_roughness_metallic_ao.png",
                    Texture.WrapMode.CLAMP_TO_EDGE,
                    Texture.ColorFormat.LINEAR
                )
            virtualObjectMesh = Mesh.createFromAsset(render, "models/pawn.obj")
            virtualObjectShader =
                Shader.createFromAssets(
                    render,
                    "shaders/environmental_hdr.vert",
                    "shaders/environmental_hdr.frag",
                    mapOf("NUMBER_OF_MIPMAP_LEVELS" to cubemapFilter.numberOfMipmapLevels.toString())
                )
                    .setTexture("u_AlbedoTexture", virtualObjectAlbedoTexture)
                    .setTexture(
                        "u_RoughnessMetallicAmbientOcclusionTexture",
                        virtualObjectPbrTexture
                    )
                    .setTexture("u_Cubemap", cubemapFilter.filteredCubemapTexture)
                    .setTexture("u_DfgTexture", dfgTexture)

            // Distance label shader and quad
            distanceShader = Shader.createFromAssets(
                render,
                "shaders/distance_label.vert",
                "shaders/distance_label.frag",
                null
            )

            val labelSize = 0.1f
            val labelHalf = labelSize / 2f
            val labelVerts = floatArrayOf(
                -labelHalf, 0f, 0f,
                labelHalf, 0f, 0f,
                labelHalf, labelSize, 0f,
                -labelHalf, labelSize, 0f
            )
            val labelBuffer: FloatBuffer = ByteBuffer
                .allocateDirect(labelVerts.size * Float.SIZE_BYTES)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer()
                .put(labelVerts)
                .apply { position(0) }
            val labelVertexBuffer = VertexBuffer(render, 3, labelBuffer)
            val labelIndices = intArrayOf(0, 1, 2, 0, 2, 3)
            val labelIndexBuffer: IntBuffer = ByteBuffer
                .allocateDirect(labelIndices.size * Int.SIZE_BYTES)
                .order(ByteOrder.nativeOrder())
                .asIntBuffer()
                .put(labelIndices)
                .apply { position(0) }

            val labelUvs = floatArrayOf(
                0f, 1f,
                1f, 1f,
                1f, 0f,
                0f, 0f
            )
            val uvBuffer = ByteBuffer
                .allocateDirect(labelUvs.size * Float.SIZE_BYTES)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer()
                .put(labelUvs)
                .apply { position(0) }
            val uvVertexBuffer = VertexBuffer(render, /*entriesPerVertex=*/2, uvBuffer)

            // Now build the mesh with both position and UV buffers
            distanceQuadMesh = Mesh(
                render,
                Mesh.PrimitiveMode.TRIANGLES,
                IndexBuffer(render, labelIndexBuffer),
                arrayOf(labelVertexBuffer, uvVertexBuffer)
            )

            // Circle shader and mesh
            innerCircleShader = Shader.createFromAssets(
                render,
                "shaders/inner_circle.vert",
                "shaders/inner_circle.frag",
                null
            )

            outerCircleShader = Shader.createFromAssets(
                render,
                "shaders/circle.vert",
                "shaders/circle.frag",
                null
            )
            val circleSegments = 32
            val circleVerts = FloatArray((circleSegments + 1) * 3)
            val angleStep = (2 * Math.PI / circleSegments).toFloat()
            for (i in 0..circleSegments) {
                val angle = i * angleStep
                circleVerts[i * 3] = (Math.cos(angle.toDouble()) * labelHalf).toFloat()
                circleVerts[i * 3 + 1] = 0f
                circleVerts[i * 3 + 2] = (Math.sin(angle.toDouble()) * labelHalf).toFloat()
            }
            val circleBuffer: FloatBuffer = ByteBuffer
                .allocateDirect(circleVerts.size * Float.SIZE_BYTES)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer()
                .put(circleVerts)
                .apply { position(0) }
            val circleVertexBuffer = VertexBuffer(render, 3, circleBuffer)
            outerCircleMesh = Mesh(
                render,
                Mesh.PrimitiveMode.LINE_LOOP,
                /*indexBuffer=*/ null,
                arrayOf(circleVertexBuffer)
            )
            val pointVerts = floatArrayOf(
                0f, 0f, 0f // X, Y, Z
            )

            val pointBuffer: FloatBuffer = ByteBuffer
                .allocateDirect(pointVerts.size * Float.SIZE_BYTES)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer()
                .put(pointVerts)
                .apply { position(0) }

            val pointVertexBuffer = VertexBuffer(render, 3, pointBuffer)

            innerCircleMesh = Mesh(
                render,
                Mesh.PrimitiveMode.POINTS,
                null,
                arrayOf(pointVertexBuffer)
            )


            circleFillShader = Shader.createFromAssets(
                render,
                "shaders/circle_fill.vert",
                "shaders/circle_fill.frag",
                null
            ).setVec4("u_Color", floatArrayOf(1f, 1f, 1f, 1f))

            val seg = 48
            val radius = labelHalf // або свій радіус
            val fanVerts = FloatArray((seg + 2) * 3) // центр + seg + повернення
            // центр
            fanVerts[0] = 0f; fanVerts[1] = 0f; fanVerts[2] = 0f
            val step = (2.0 * Math.PI / seg)
            for (i in 0..seg) {
                val a = i * step
                val x = (Math.cos(a) * radius).toFloat()
                val z = (Math.sin(a) * radius).toFloat()
                val o = (i + 1) * 3
                fanVerts[o] = x
                fanVerts[o + 1] = 0f
                fanVerts[o + 2] = z
            }
            val fanBuffer = ByteBuffer.allocateDirect(fanVerts.size * 4)
                .order(ByteOrder.nativeOrder()).asFloatBuffer().put(fanVerts).apply { position(0) }
            val fanVbo = VertexBuffer(render, 3, fanBuffer)
            // build indices for triangles: (0, i, i+1)
            val triIdx = IntArray(seg * 3)
            var k = 0
            for (i in 1..seg) {
                triIdx[k++] = 0
                triIdx[k++] = i
                triIdx[k++] = i + 1
            }
            val triIdxBuf = ByteBuffer.allocateDirect(triIdx.size * Int.SIZE_BYTES)
                .order(ByteOrder.nativeOrder()).asIntBuffer().put(triIdx).apply { position(0) }
            val triIbo = IndexBuffer(render, triIdxBuf)
            circleFillMesh = Mesh(render, Mesh.PrimitiveMode.TRIANGLES, triIbo, arrayOf(fanVbo))


            // --- Small center dot (filled circle) ---
            val dotSeg = 48
            val dotRadius = radius * 0.5f // половина радіуса зовнішнього круга
            val dotVerts = FloatArray((dotSeg + 2) * 3)
// центр
            dotVerts[0] = 0f; dotVerts[1] = 0f; dotVerts[2] = 0f
            val dotStep = (2.0 * Math.PI / dotSeg)
            for (i in 0..dotSeg) {
                val a = i * dotStep
                val x = (Math.cos(a) * dotRadius).toFloat()
                val z = (Math.sin(a) * dotRadius).toFloat()
                val o = (i + 1) * 3
                dotVerts[o] = x
                dotVerts[o + 1] = 0f
                dotVerts[o + 2] = z
            }
            val dotBuf = ByteBuffer.allocateDirect(dotVerts.size * 4)
                .order(ByteOrder.nativeOrder()).asFloatBuffer()
                .put(dotVerts).apply { position(0) }
            val dotVbo = VertexBuffer(render, 3, dotBuf)
// індекси для трикутників (0, i, i+1)
            val dotIdx = IntArray(dotSeg * 3)
            var dk = 0
            for (i in 1..dotSeg) {
                dotIdx[dk++] = 0
                dotIdx[dk++] = i
                dotIdx[dk++] = i + 1
            }
            val dotIdxBuf = ByteBuffer.allocateDirect(dotIdx.size * Int.SIZE_BYTES)
                .order(ByteOrder.nativeOrder()).asIntBuffer()
                .put(dotIdx).apply { position(0) }
            val dotIbo = IndexBuffer(render, dotIdxBuf)
            dotFillMesh = Mesh(render, Mesh.PrimitiveMode.TRIANGLES, dotIbo, arrayOf(dotVbo))

            // --- Simple line (two points) to connect two anchors ---
            // Requires assets/shaders/line.vert and assets/shaders/line.frag (see comments below)
            lineShader = Shader.createFromAssets(
                render,
                "shaders/line.vert",
                "shaders/line.frag",
                null
            )

            // Allocate a dynamic FloatBuffer for two 3D points (x0,y0,z0, x1,y1,z1)
            val initialLine = floatArrayOf(
                0f, 0f, 0f,
                0f, 0f, 0f
            )
            lineFloatBuffer = ByteBuffer
                .allocateDirect(initialLine.size * Float.SIZE_BYTES)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer()
                .put(initialLine)
                .apply { position(0) }

            lineVertexBuffer = VertexBuffer(render, /*entriesPerVertex=*/3, lineFloatBuffer)
            lineMesh = Mesh(
                render,
                Mesh.PrimitiveMode.LINES,
                /*indexBuffer=*/ null,
                arrayOf(lineVertexBuffer)
            )
        } catch (e: IOException) {
            Log.e(TAG, "Failed to read a required asset file", e)
            showError("Failed to read a required asset file: $e")
        }
    }

    private val EPS = 0.003f // ~3 мм над площиною

    /** Translate model along the plane normal (anchor's local +Y) by `meters`. */
    private fun offsetAlongPlaneNormal(pose: Pose, model: FloatArray, meters: Float) {
        val n = FloatArray(3)
        pose.getTransformedAxis(1, 1f, n, 0) // 1 = Y axis in pose local space
        model[12] += n[0] * meters
        model[13] += n[1] * meters
        model[14] += n[2] * meters
    }

    private fun drawStickerOnPlane(
        anchor: Anchor,
        mesh: Mesh,
        shader: Shader,
        scale: Float,
        viewMatrix: FloatArray,
        projMatrix: FloatArray,
        extraLiftMeters: Float = 0f,
        tmpModel: FloatArray = FloatArray(16),
        tmpMV: FloatArray = FloatArray(16),
        tmpMVP: FloatArray = FloatArray(16)
    ) {
        // 1) Model matrix = anchor pose (it already contains plane rotation)
        anchor.pose.toMatrix(tmpModel, 0)

        // 2) Shift along plane normal = EPS + extraLiftMeters
        offsetAlongPlaneNormal(anchor.pose, tmpModel, EPS + extraLiftMeters)

        // 3) Scale (to set the actual size of the sticker)
        val s = FloatArray(16)
        Matrix.setIdentityM(s, 0)
        Matrix.scaleM(s, 0, scale, scale, scale)
        // postMultiply: model = model * scale
        Matrix.multiplyMM(tmpModel, 0, tmpModel, 0, s, 0)

        // 4) MVP
        Matrix.multiplyMM(tmpMV, 0, viewMatrix, 0, tmpModel, 0)
        Matrix.multiplyMM(tmpMVP, 0, projMatrix, 0, tmpMV, 0)

        shader.setMat4("u_MVP", tmpMVP)
        render.draw(mesh, shader)
    }

    override fun onSurfaceChanged(render: SampleRender, width: Int, height: Int) {
        displayRotationHelper.onSurfaceChanged(width, height)
        virtualSceneFramebuffer.resize(width, height)
        viewWidth = width
        viewHeight = height
    }

    override fun onDrawFrame(render: SampleRender) {
        val session = session ?: return

        // Texture names should only be set once on a GL thread unless they change. This is done during
        // onDrawFrame rather than onSurfaceCreated since the session is not guaranteed to have been
        // initialized during the execution of onSurfaceCreated.
        if (!hasSetTextureNames) {
            session.setCameraTextureNames(intArrayOf(backgroundRenderer.cameraColorTexture.textureId))
            hasSetTextureNames = true
        }

        // -- Update per-frame state

        // Notify ARCore session that the view size changed so that the perspective matrix and
        // the video background can be properly adjusted.
        displayRotationHelper.updateSessionIfNeeded(session)

        // Obtain the current frame from ARSession. When the configuration is set to
        // UpdateMode.BLOCKING (it is by default), this will throttle the rendering to the
        // camera framerate.
        val frame =
            try {
                session.update()
            } catch (e: CameraNotAvailableException) {
                Log.e(TAG, "Camera not available during onDrawFrame", e)
                showError("Camera not available. Try restarting the app.")
                return
            }


        val camera = frame.camera

        // Update BackgroundRenderer state to match the depth settings.
        try {
            backgroundRenderer.setUseDepthVisualization(
                render,
                activity.depthSettings.depthColorVisualizationEnabled()
            )
            backgroundRenderer.setUseOcclusion(
                render,
                activity.depthSettings.useDepthForOcclusion()
            )
        } catch (e: IOException) {
            Log.e(TAG, "Failed to read a required asset file", e)
            showError("Failed to read a required asset file: $e")
            return
        }

        // BackgroundRenderer.updateDisplayGeometry must be called every frame to update the coordinates
        // used to draw the background camera image.
        backgroundRenderer.updateDisplayGeometry(frame)
        val shouldGetDepthImage =
            activity.depthSettings.useDepthForOcclusion() ||
                    activity.depthSettings.depthColorVisualizationEnabled()
        if (camera.trackingState == TrackingState.TRACKING && shouldGetDepthImage) {
            try {
                val depthImage = frame.acquireDepthImage16Bits()
                backgroundRenderer.updateCameraDepthTexture(depthImage)
                depthImage.close()
            } catch (e: NotYetAvailableException) {
                // This normally means that depth data is not available yet. This is normal so we will not
                // spam the logcat with this.
            }
        }

        // Handle one tap per frame.
        handleTap(frame, camera)

        // Keep the screen unlocked while tracking, but allow it to lock when tracking stops.
        trackingStateHelper.updateKeepScreenOnFlag(camera.trackingState)

        // Show a message based on whether tracking has failed, if planes are detected, and if the user
        // has placed any objects.
        val message: String? =
            when {
                camera.trackingState == TrackingState.PAUSED &&
                        camera.trackingFailureReason == TrackingFailureReason.NONE ->
                    activity.getString(R.string.searching_planes)

                camera.trackingState == TrackingState.PAUSED ->
                    TrackingStateHelper.getTrackingFailureReasonString(camera)

                session.hasTrackingPlane() && wrappedAnchors.isEmpty() ->
                    activity.getString(R.string.waiting_taps)

                session.hasTrackingPlane() && wrappedAnchors.isNotEmpty() -> null
                else -> activity.getString(R.string.searching_planes)
            }
        if (message == null) {
            activity.view.snackbarHelper.hide(activity)
        } else {
            activity.view.snackbarHelper.showMessage(activity, message)
        }

        // -- Draw background
        if (frame.timestamp != 0L) {
            // Suppress rendering if the camera did not produce the first frame yet. This is to avoid
            // drawing possible leftover data from previous sessions if the texture is reused.
            backgroundRenderer.drawBackground(render)
        }

        // If not tracking, don't draw 3D objects.
        if (camera.trackingState == TrackingState.PAUSED) {
            return
        }

        // -- Draw non-occluded virtual objects (planes, point cloud)

        // Get projection matrix.
        camera.getProjectionMatrix(projectionMatrix, 0, Z_NEAR, Z_FAR)

        // Get camera matrix and draw.
        camera.getViewMatrix(viewMatrix, 0)

        // Draw center reticle (now with valid view/projection matrices)
//        val centerX = viewWidth / 2f
//        val centerY = viewHeight / 2f
//        val hits = frame.hitTest(centerX, centerY)
//        val hitPose = hits.firstOrNull { hit ->
//            val trackable = hit.trackable
//            (trackable is Plane && trackable.isPoseInPolygon(hit.hitPose)) ||
//            (trackable is InstantPlacementPoint)
//        }?.hitPose
//        if (hitPose != null) {
//
//            drawReticleAtPose(render, hitPose)
//        }

        try {
            frame.acquirePointCloud().use { pointCloud ->
                if (pointCloud.timestamp > lastPointCloudTimestamp) {
                    pointCloudVertexBuffer.set(pointCloud.points)
                    lastPointCloudTimestamp = pointCloud.timestamp
                }
                Matrix.multiplyMM(modelViewProjectionMatrix, 0, projectionMatrix, 0, viewMatrix, 0)
                pointCloudShader.setMat4("u_ModelViewProjection", modelViewProjectionMatrix)
                render.draw(pointCloudMesh, pointCloudShader)
            }
        } catch (e: DeadlineExceededException) {
            // Skip point cloud rendering this frame if ARCore deadline exceeded
            Log.w(TAG, "PointCloud acquisition took too long: ${e.message}")
        }

        // Visualize planes.
        planeRenderer.drawPlanes(
            render,
            session.getAllTrackables<Plane>(Plane::class.java),
            camera.displayOrientedPose,
            projectionMatrix
        )

        // -- Draw occluded virtual objects

        // Update lighting parameters in the shader
        updateLightEstimation(frame.lightEstimate, viewMatrix)
        when (selectedMode) {
            MeasurementMode.Camera -> {
                // Visualize anchors created by touch.
                render.clear(virtualSceneFramebuffer, 0f, 0f, 0f, 0f)
                for ((anchor, trackable) in
                    wrappedAnchors.filter { it.anchor.trackingState == TrackingState.TRACKING }) {
                    // --- build MVP for distance label, lifted above the dot ---
                    anchor.pose.toMatrix(modelMatrix, 0)
                    val camPose = frame.camera.pose
                    val objPose = anchor.pose
                    val dx = objPose.tx() - camPose.tx()
                    val dy = objPose.ty() - camPose.ty()
                    val dz = objPose.tz() - camPose.tz()
                    val distCm = kotlin.math.sqrt(dx * dx + dy * dy + dz * dz)

                    val modelForLabel = modelMatrix.copyOf()
                    val dotRadiusM = 0.03f
                    val marginM = 0.01f
                    offsetAlongPlaneNormal(anchor.pose, modelForLabel, dotRadiusM + marginM)
                    Matrix.multiplyMM(modelViewMatrix, 0, viewMatrix, 0, modelForLabel, 0)
                    Matrix.multiplyMM(modelViewProjectionMatrix, 0, projectionMatrix, 0, modelViewMatrix, 0)

                    updateDistanceTexture(distCm)
                    distanceShader.setMat4("u_MVP", modelViewProjectionMatrix)
                    distanceShader.setTexture("u_Texture", distanceTexture!!)
                    render.draw(distanceQuadMesh, distanceShader)
                    // --- build MVP for circle/dot flush on plane (no extra lift beyond EPS handled in drawStickerOnPlane) ---
                    anchor.pose.toMatrix(modelMatrix, 0)
                    Matrix.multiplyMM(modelViewMatrix, 0, viewMatrix, 0, modelMatrix, 0)
                    Matrix.multiplyMM(modelViewProjectionMatrix, 0, projectionMatrix, 0, modelViewMatrix, 0)
                    outerCircleShader.setMat4("u_MVP", modelViewProjectionMatrix)
                    render.draw(outerCircleMesh, outerCircleShader)
                    drawStickerOnPlane(
                        anchor = anchor,
                        mesh = innerCircleMesh,
                        shader = innerCircleShader,
                        scale = 1f,
                        viewMatrix = viewMatrix,
                        projMatrix = projectionMatrix,
                        extraLiftMeters = 0.03f
                    )
                }
                measureDistanceFromCamera(frame)
            }

            MeasurementMode.TwoPoints -> {
                render.clear(virtualSceneFramebuffer, 0f, 0f, 0f, 0f)

                for ((anchor, _) in wrappedAnchors.filter { it.anchor.trackingState == TrackingState.TRACKING }) {
                    val pose = anchor.pose
                    pose.toMatrix(modelMatrix, 0)
                    Matrix.multiplyMM(modelViewMatrix, 0, viewMatrix, 0, modelMatrix, 0)
                    Matrix.multiplyMM(
                        modelViewProjectionMatrix,
                        0,
                        projectionMatrix,
                        0,
                        modelViewMatrix,
                        0
                    )
                    drawStickerOnPlane(
                        anchor = anchor,
                        mesh = innerCircleMesh,
                        shader = innerCircleShader,
                        scale = 1f,
                        viewMatrix = viewMatrix,
                        projMatrix = projectionMatrix,
                        extraLiftMeters = 0.03f
                    )

                    Matrix.multiplyMM(modelViewMatrix, 0, viewMatrix, 0, modelMatrix, 0)
                    Matrix.multiplyMM(modelViewProjectionMatrix, 0, projectionMatrix, 0, modelViewMatrix, 0)
                    outerCircleShader.setMat4("u_MVP", modelViewProjectionMatrix)
                    render.draw(outerCircleMesh, outerCircleShader)
                }

                // Draw the connecting line between the two anchors (if present)
                if (wrappedAnchors.size >= 2) {
                    val a = wrappedAnchors[0].anchor.pose
                    val b = wrappedAnchors[1].anchor.pose
                    val dx = b.tx() - a.tx()
                    val dy = b.ty() - a.ty()
                    val dz = b.tz() - a.tz()
                    val distMeters = kotlin.math.sqrt(dx * dx + dy * dy + dz * dz)
                    // --- Update and draw the white line segment between the two anchors ---
                    // Offset both endpoints +0.03m along each pose's +Y (plane normal), same as inner dot
                    val na = FloatArray(3)
                    val nb = FloatArray(3)
                    a.getTransformedAxis(1, 0.03f, na, 0) // 1 = Y axis
                    b.getTransformedAxis(1, 0.03f, nb, 0)

                    val ax = a.tx() + na[0]
                    val ay = a.ty() + na[1]
                    val az = a.tz() + na[2]
                    val bx = b.tx() + nb[0]
                    val by = b.ty() + nb[1]
                    val bz = b.tz() + nb[2]

                    // Write the lifted endpoints into the dynamic buffer
                    lineFloatBuffer.rewind()
                    lineFloatBuffer.put(ax).put(ay).put(az)
                    lineFloatBuffer.put(bx).put(by).put(bz)
                    lineFloatBuffer.rewind()
                    lineVertexBuffer.set(lineFloatBuffer)

                    // For the line we use model = identity; so MVP = P * V
                    Matrix.multiplyMM(vpMatrix, 0, projectionMatrix, 0, viewMatrix, 0)
                    lineShader.setMat4("u_VP", vpMatrix)
                    render.draw(lineMesh, lineShader)

                    val midT = floatArrayOf(
                        (a.tx() + b.tx()) / 2f,
                        (a.ty() + b.ty()) / 2f,
                        (a.tz() + b.tz()) / 2f
                    )
                    val midQ = floatArrayOf(a.qx(), a.qy(), a.qz(), a.qw())
                    val midPose = com.google.ar.core.Pose(midT, midQ)
                    midPose.toMatrix(modelMatrix, 0)
                    Matrix.multiplyMM(modelViewMatrix, 0, viewMatrix, 0, modelMatrix, 0)
                    Matrix.multiplyMM(
                        modelViewProjectionMatrix,
                        0,
                        projectionMatrix,
                        0,
                        modelViewMatrix,
                        0
                    )
//                    val modelForLabel = modelMatrix.copyOf()
//                    val dotRadiusM = 0.05f
//                    val marginM = 0.01f
//                    offsetAlongPlaneNormal(midPose, modelForLabel, dotRadiusM + marginM)
//                    Matrix.multiplyMM(modelViewMatrix, 0, viewMatrix, 0, modelForLabel, 0)
//                    Matrix.multiplyMM(modelViewProjectionMatrix, 0, projectionMatrix, 0, modelViewMatrix, 0)

                    updateDistanceTexture(distMeters)

                    distanceShader.setMat4("u_MVP", modelViewProjectionMatrix)
                    distanceShader.setTexture("u_Texture", distanceTexture!!)
                    render.draw(distanceQuadMesh, distanceShader)
                }
            }

            MeasurementMode.SeveralPoints -> {

            }
        }


        // Compose the virtual scene with the background.
        backgroundRenderer.drawVirtualScene(render, virtualSceneFramebuffer, Z_NEAR, Z_FAR)
    }

    /** Checks if we detected at least one plane. */
    private fun Session.hasTrackingPlane() =
        getAllTrackables(Plane::class.java).any { it.trackingState == TrackingState.TRACKING }

    /** Update state based on the current frame's light estimation. */
    private fun updateLightEstimation(lightEstimate: LightEstimate, viewMatrix: FloatArray) {
        if (lightEstimate.state != LightEstimate.State.VALID) {
            virtualObjectShader.setBool("u_LightEstimateIsValid", false)
            return
        }
        virtualObjectShader.setBool("u_LightEstimateIsValid", true)
        Matrix.invertM(viewInverseMatrix, 0, viewMatrix, 0)
        virtualObjectShader.setMat4("u_ViewInverse", viewInverseMatrix)
        updateMainLight(
            lightEstimate.environmentalHdrMainLightDirection,
            lightEstimate.environmentalHdrMainLightIntensity,
            viewMatrix
        )
        updateSphericalHarmonicsCoefficients(lightEstimate.environmentalHdrAmbientSphericalHarmonics)
        cubemapFilter.update(lightEstimate.acquireEnvironmentalHdrCubeMap())
    }

    private fun updateMainLight(
        direction: FloatArray,
        intensity: FloatArray,
        viewMatrix: FloatArray
    ) {
        // We need the direction in a vec4 with 0.0 as the final component to transform it to view space
        worldLightDirection[0] = direction[0]
        worldLightDirection[1] = direction[1]
        worldLightDirection[2] = direction[2]
        Matrix.multiplyMV(viewLightDirection, 0, viewMatrix, 0, worldLightDirection, 0)
        virtualObjectShader.setVec4("u_ViewLightDirection", viewLightDirection)
        virtualObjectShader.setVec3("u_LightIntensity", intensity)
    }

    private fun updateSphericalHarmonicsCoefficients(coefficients: FloatArray) {
        // Pre-multiply the spherical harmonics coefficients before passing them to the shader. The
        // constants in sphericalHarmonicFactors were derived from three terms:
        //
        // 1. The normalized spherical harmonics basis functions (y_lm)
        //
        // 2. The lambertian diffuse BRDF factor (1/pi)
        //
        // 3. A <cos> convolution. This is done to so that the resulting function outputs the irradiance
        // of all incoming light over a hemisphere for a given surface normal, which is what the shader
        // (environmental_hdr.frag) expects.
        //
        // You can read more details about the math here:
        // https://google.github.io/filament/Filament.html#annex/sphericalharmonics
        require(coefficients.size == 9 * 3) {
            "The given coefficients array must be of length 27 (3 components per 9 coefficients"
        }

        // Apply each factor to every component of each coefficient
        for (i in 0 until 9 * 3) {
            sphericalHarmonicsCoefficients[i] = coefficients[i] * sphericalHarmonicFactors[i / 3]
        }
        virtualObjectShader.setVec3Array(
            "u_SphericalHarmonicsCoefficients",
            sphericalHarmonicsCoefficients
        )
    }

    /// We must filter the raw hit-test results and create anchors only on valid surfaces.
    /// Otherwise ARCore can throw FatalException from HitResult.createAnchor() when the pose
    /// is outside a plane polygon, behind the plane, or refers to a transient/unsupported trackable.
    private fun handleTap(frame: Frame, camera: Camera) {
        if (camera.trackingState != TrackingState.TRACKING) return
        val tap = activity.view.tapHelper.poll() ?: return

        // Collect hits for this tap
        val hitResultList = frame.hitTest(tap)

        // Keep only safe/valid hits
        fun isValidHit(hit: HitResult): Boolean {
            val trackable = hit.trackable
            return when (trackable) {
                is Plane -> {
                    // Must be inside polygon and in front of the plane (positive distance)
                    trackable.isPoseInPolygon(hit.hitPose) &&
                            PlaneRenderer.calculateDistanceToPlane(hit.hitPose, camera.pose) > 0
                }
                // Accept oriented points (feature points) and depth points
                is com.google.ar.core.Point -> {
                    trackable.orientationMode == com.google.ar.core.Point.OrientationMode.ESTIMATED_SURFACE_NORMAL
                }
                is com.google.ar.core.DepthPoint -> true
                is com.google.ar.core.InstantPlacementPoint -> true
                else -> false
            }
        }

        val firstValidHit = hitResultList.firstOrNull { isValidHit(it) } ?: return

        when (selectedMode) {
            MeasurementMode.Camera -> {
                placeAnchorCameraMode(firstValidHit)
            }
            MeasurementMode.TwoPoints -> {
                placeAnchorTwoPointsMode(firstValidHit)
            }
            MeasurementMode.SeveralPoints -> {
                // TODO
            }
        }
    }

    private fun showError(errorMessage: String) =
        activity.view.snackbarHelper.showError(activity, errorMessage)


    fun updateDistanceTexture(distCm: Float) {
        val texW = 512
        val texH = 256
        val bmp = createBitmap(texW, texH)
        val canvas = Canvas(bmp)

        val paint = Paint().apply {
            isAntiAlias = true
            isLinearText = true
            textSize = texH * 0.5f
            textAlign = Paint.Align.CENTER
            color = Color.WHITE
            typeface = Typeface.create(Typeface.SANS_SERIF, Typeface.NORMAL)
        }

        val bgPaint = Paint().apply {
            isAntiAlias = true
            color = Color.argb(200, 0, 0, 0)
        }
        canvas.drawRoundRect(
            0f, texH * 0.25f, texW.toFloat(), texH * 0.75f, 20f, 20f, bgPaint
        )

        canvas.drawText(
            "%.2f m".format(distCm),
            texW * 0.5f,
            texH * 0.6f,
            paint
        )

        distanceTexture = Texture.createFromBitmap(
            render,
            bmp,
            Texture.WrapMode.CLAMP_TO_EDGE,
            Texture.ColorFormat.SRGB
        )
    }

    private fun placeAnchorCameraMode(hitResult: HitResult) {
        try {
            // Ensure we always keep a single anchor in this mode
            anchor?.detach()
            wrappedAnchors.forEach { it.anchor.detach() }
            wrappedAnchors.clear()

            val newAnchor = hitResult.createAnchor()
            anchor = newAnchor
            wrappedAnchors.add(WrappedAnchor(newAnchor, hitResult.trackable!!))
            Log.d(TAG, "Anchor placed at: ${newAnchor.pose}")
        } catch (t: Throwable) {
            // Some invalid hits can still slip through on certain devices / frames; fail gracefully
            Log.e(TAG, "Failed to create anchor from hit: ${t.message}", t)
        }
    }

    private fun placeAnchorTwoPointsMode(hitResult: HitResult) {
        try {
            if (anchorPair == null) {
                val firstAnchor = hitResult.createAnchor()
                wrappedAnchors.add(WrappedAnchor(firstAnchor, hitResult.trackable))
                anchorPair = Pair(firstAnchor, null)
            } else if (anchorPair?.second == null) {
                val secondAnchor = hitResult.createAnchor()
                wrappedAnchors.add(WrappedAnchor(secondAnchor, hitResult.trackable))
                anchorPair = anchorPair?.copy(second = secondAnchor)
            } else {
                anchorPair?.apply {
                    first.detach()
                    second?.detach()
                }
                anchorPair = null
                wrappedAnchors.clear()
            }
        } catch (t: Throwable) {
            Log.e(TAG, "Failed to create anchor (TwoPoints): ${t.message}", t)
        }
    }

    private fun measureDistanceFromCamera(frame: Frame) {
        if (wrappedAnchors.isNotEmpty()) {
            val camPose = frame.camera.pose
            val objPose = wrappedAnchors[0].anchor.pose

            val dx = objPose.tx() - camPose.tx()
            val dy = objPose.ty() - camPose.ty()
            val dz = objPose.tz() - camPose.tz()
            val distMeters = kotlin.math.sqrt(dx * dx + dy * dy + dz * dz)

            Log.d(TAG, "Distance to first anchor: %.2f m".format(distMeters))
        }
    }

    private fun calculateDistance(x: Float, y: Float, z: Float): Float {
        return sqrt(x.pow(2) + y.pow(2) + z.pow(2))
    }

    private fun calculateDistance(objectPose0: Pose, objectPose1: Pose): Float {
        return calculateDistance(
            objectPose0.tx() - objectPose1.tx(),
            objectPose0.ty() - objectPose1.ty(),
            objectPose0.tz() - objectPose1.tz()
        )
    }


    private fun changeUnit(distanceMeter: Float, unit: String): Float {
        return when (unit) {
            "cm" -> distanceMeter * 100
            "mm" -> distanceMeter * 1000
            else -> distanceMeter
        }
    }

    private var anchor: Anchor? = null

    private var anchorPair: Pair<Anchor, Anchor?>? = null


    /**
     * Fully reset renderer state when switching modes.
     * Detaches all anchors, clears lists/caches and resets per-frame state
     * so the next frame starts "from scratch".
     */
    private fun resetForModeChange() {
        // Detach and clear all placed anchors
        try {
            wrappedAnchors.forEach { it.anchor.detach() }
        } catch (_: Exception) { /* ignore */
        }
        wrappedAnchors.clear()

        // Detach single reference anchor if present
        try {
            anchor?.detach()
        } catch (_: Exception) { /* ignore */
        }
        anchor = null

        // Drop dynamic textures so they can be recreated fresh
        distanceTexture = null

        // Reset transient rendering state
        lastPointCloudTimestamp = 0L

        // Optionally clear the offscreen framebuffer to avoid ghosting
        if (this::virtualSceneFramebuffer.isInitialized) {
            try {
                render.clear(virtualSceneFramebuffer, 0f, 0f, 0f, 0f)
            } catch (_: Exception) { /* ignore */
            }
        }

        Log.d(TAG, "Renderer state reset for mode change")
    }

    fun setSelectedMeasurementMode(mode: MeasurementMode) {
        if (mode == selectedMode) return
        selectedMode = mode
        resetForModeChange()
        Log.d("TAG", "selectedmode -> $mode ");
    }
    fun clearLastAnchor() {

    }
    fun clearAllAnchors() {


    }
}

/**
 * Associates an Anchor with the trackable it was attached to. This is used to be able to check
 * whether or not an Anchor originally was attached to an {@link InstantPlacementPoint}.
 */
private data class WrappedAnchor(
    val anchor: Anchor,
    val trackable: Trackable,
)
