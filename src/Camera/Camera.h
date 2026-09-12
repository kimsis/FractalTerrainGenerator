#pragma once
#include "glm/glm.hpp"

class Camera {
   public:
    /*!
     * Camera constructor
     * @param	fov		            The field of view to be used for this camera's projection matrix
     * @param	ar		            The aspect ratio to be used for this camera's projection matrix
     * @param	near_plane_dist		The distance to the near plane to be used for this camera's projection matrix
     * @param	far_plane_dist		The distance to the far plane to be used for this camera's projection matrix
     */
    Camera(float fov, float ar, float near_plane_dist, float far_plane_dist);

    /*! Builds the view matrix for this camera. Each camera type computes this differently. */
    virtual glm::mat4 getViewMatrix() const = 0;

    /*! projection * getViewMatrix() — identical formula for both camera types regardless of how
     *  each computes its own view matrix, so shared and non-virtual here, matching getRight(). */
    glm::mat4 getViewProjectionMatrix() const;

    /*! The direction the camera is currently facing. Each camera type derives this differently. */
    virtual glm::vec3 getForward() const = 0;

    /*! Camera-relative "right" axis. */
    glm::vec3 getRight() const;

    /*! By value, not by reference, and virtual: TrackballCamera doesn't store its true position in
     *  `position` (see its override) — it's derived from target/radius/yaw/pitch and computed
     *  fresh, so there's no persistent member for a reference to safely point at. */
    virtual glm::vec3 getPosition() const;
    float getYaw() const;
    float getPitch() const;
    float getSpeed() const;

    /*! Change the aspect ratio when the window is resized */
    void setAspectRatio(float aspect_ratio);

    void computeProjectionMatrix();

    /*! Change the speed at which the camera moves */
    void setSpeed(float newSpeed);

    /*! Applies a mouse-drag delta to yaw/pitch, clamping pitch to avoid gimbal lock at the poles.
     *  Shared between both camera types since the input handling is identical either way.
     *  @param yawDelta     intermediate yaw value in radians
     *  @param pitchDelta   intermediate pitch value in radians
     */
    void rotate(float yawDelta, float pitchDelta, float minPitch = glm::radians(-89.9f), float maxPitch = glm::radians(89.9f));

    /*! Moves the camera's position by a world-space delta (scaled by speed). */
    virtual void translate(const glm::vec3& delta);

    /*! World up vector, unchangable. Shared by both camera types: FlyCamera derives its own
     *  right axis from this and its forward direction; TrackballCamera does the same for panning. */
    static constexpr glm::vec3 up{0.0f, 0.0f, 1.0f};

   protected:
    /*! Rotation around the vertical line */
    float yaw = 0.0f;

    /*! Rotation around the lateral line */
    float pitch = 0.0f;

    /*! 3D position of the camera, in world coordinates */
    glm::vec3 position{0.0f};

    /*! Camera translation speed */
    float speed = 1.0f;

    float field_of_view;
    float aspect_ratio;
    float near_plane_distance;
    float far_plane_distance;

    glm::mat4 projection_matrix;
};

class TrackballCamera : public Camera {
   public:
    using Camera::Camera;

    /*! Converts an existing Camera (typically a FlyCamera) into a TrackballCamera, carrying
     *  position/yaw/pitch/speed/projection across rather than resetting them — for switching
     *  movement styles mid-session without the camera jumping. `target` must be supplied
     *  externally (e.g. via a raycast against the terrain along the current look direction),
     *  since it's the one piece of state a free-fly camera never had. */
    TrackballCamera(const Camera& fromCamera, const glm::vec3& target);

    glm::mat4 getViewMatrix() const override;

    /*! Overridden because the inherited `position` member is NOT this camera's true position —
     *  see computeOrbitPosition()'s comment for why. */
    glm::vec3 getPosition() const override;

    /*! Pans the camera: shifts both position and target by the same world-space delta, so the
     *  orbit itself (radius, yaw/pitch relative to target) is preserved. */
    void translate(const glm::vec3& delta) override;

    /*! Direction from the camera to its target, derived fresh every call — same derived-not-cached
     *  reasoning as FlyCamera::getForward(), and built on computeOrbitPosition() rather than the
     *  (unreliable) inherited `position` member. */
    glm::vec3 getForward() const override;

    /*! Camera-relative "up" axis. NOT the same as world `up` in general — getForward() isn't
     *  necessarily horizontal — so this is recomputed via cross(right, forward) to stay
     *  perpendicular to both, rather than reusing world `up` directly. */
    glm::vec3 getUp() const;

   private:
    /*! The camera's actual current position, derived fresh from target/radius/yaw/pitch every
     *  call. This orbit is not stored in the inherited `position` member at all: translate()
     *  moves `target` (and, harmlessly, `position`) but rotate() only ever touches yaw/pitch —
     *  nothing keeps `position` in sync as the orbit angles change. Using `position` directly
     *  anywhere in this class (as getForward() once did) silently goes stale the moment the
     *  camera orbits, and is flat-out NaN at the default-constructed state (position == target
     *  == origin, so target - position is a zero-length vector). computeOrbitPosition() is the
     *  single source of truth getViewMatrix(), getForward(), and getPosition() all share instead. */
    glm::vec3 computeOrbitPosition() const;

    /*! 3D position at which the camera is locked onto, in world coordinates */
    glm::vec3 target{0.0f};

    /*! The vector direction from the camera position to the target */
    glm::vec3 lookAt{0.0f};

    /*! The length between the target and camera position*/
    float radius = 1.0f;
};

class FlyCamera : public Camera {
   public:
    using Camera::Camera;

    /*! Converts an existing Camera (typically a TrackballCamera) into a FlyCamera, carrying
     *  position/yaw/pitch/speed/projection across. No extra state to supply here — unlike the
     *  reverse direction, FlyCamera doesn't need anything beyond what Camera already holds. */
    explicit FlyCamera(const Camera& fromCamera);

    glm::mat4 getViewMatrix() const override;

    /*! The direction the camera is currently facing, derived fresh from yaw/pitch every call */
    glm::vec3 getForward() const override;

    /*! Moves the camera along its current facing/strafing direction, scaled by elapsed frame time
     *  (dt) — speed itself is applied inside translate(), not here. */
    void moveForward(float dt);
    void moveBackward(float dt);
    void moveRight(float dt);
    void moveLeft(float dt);

    /*! Moves straight along world `up`, not the camera-relative up used elsewhere — pitching the
     *  view doesn't tilt which way "higher" means for these two. */
    void moveUp(float dt);
    void moveDown(float dt);
};