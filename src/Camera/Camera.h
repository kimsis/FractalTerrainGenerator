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

    /*! Builds the view matrix for this camera. */
    virtual glm::mat4 getViewMatrix() const = 0;

    /*! projection * getViewMatrix(). */
    glm::mat4 getViewProjectionMatrix() const;

    /*! The direction the camera is currently facing. */
    virtual glm::vec3 getForward() const = 0;

    /*! Camera-relative "right" axis. */
    glm::vec3 getRight() const;

    /*! The camera's current world-space position. */
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
     *  @param yawDelta     intermediate yaw value in radians
     *  @param pitchDelta   intermediate pitch value in radians
     */
    void rotate(float yawDelta, float pitchDelta, float minPitch = glm::radians(-89.9f), float maxPitch = glm::radians(89.9f));

    /*! Moves the camera's position by a world-space delta (scaled by speed). */
    virtual void translate(const glm::vec3& delta);

    /*! World up vector, unchangable. */
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
    static const float kScrollSensitivity = 0.5f;
    static const float kPanSensitivity = 0.01f;
    using Camera::Camera;

    /*! Converts an existing Camera into a TrackballCamera, carrying position/yaw/pitch/speed/
     *  projection across.
     *  @param target   The point the camera orbits around. */
    TrackballCamera(const Camera& fromCamera, const glm::vec3& target);

    glm::mat4 getViewMatrix() const override;

    /*! The camera's current world-space position. */
    glm::vec3 getPosition() const override;

    /*! Pans the camera: shifts both position and target by the same world-space delta. */
    void translate(const glm::vec3& delta) override;

    /*! Direction from the camera to its target. */
    glm::vec3 getForward() const override;

    /*! Camera-relative "up" axis (not the same as world `up` in general). */
    glm::vec3 getUp() const;

    /*! Changes the distance from target, without moving target itself. */
    void zoom(float delta);

   private:
    /*! The camera's current position on the orbit, derived from target/radius/yaw/pitch. */
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

    /*! Converts an existing Camera into a FlyCamera, carrying position/yaw/pitch/speed/projection
     *  across. */
    explicit FlyCamera(const Camera& fromCamera);

    glm::mat4 getViewMatrix() const override;

    /*! The direction the camera is currently facing. */
    glm::vec3 getForward() const override;

    /*! Moves the camera along its current facing/strafing direction, scaled by elapsed frame time
     *  (dt). Speed is applied inside translate(), not here. */
    void moveForward(float dt);
    void moveBackward(float dt);
    void moveRight(float dt);
    void moveLeft(float dt);

    /*! Moves straight along world `up`, not the camera-relative up used elsewhere. */
    void moveUp(float dt);
    void moveDown(float dt);
};
