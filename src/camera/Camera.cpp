#include "Camera.h"

#include <VulkanLaunchpad.h>

#include <glm/gtc/matrix_transform.hpp>

// ----- Camera -----

Camera::Camera(float fov, float ar, float near_plane_dist, float far_plane_dist) {
    field_of_view = fov;
    aspect_ratio = ar;
    near_plane_distance = near_plane_dist;
    far_plane_distance = far_plane_dist;
    computeProjectionMatrix();
}

glm::vec3 Camera::getRight() const { return glm::normalize(glm::cross(getForward(), up)); }

glm::mat4 Camera::getViewProjectionMatrix() const { return projection_matrix * getViewMatrix(); }

glm::vec3 Camera::getPosition() const { return position; }

float Camera::getYaw() const { return yaw; }

float Camera::getPitch() const { return pitch; }

float Camera::getSpeed() const { return speed; }

void Camera::computeProjectionMatrix() {
    projection_matrix = vklCreatePerspectiveProjectionMatrix(glm::radians(field_of_view), aspect_ratio, near_plane_distance, far_plane_distance);
}

void Camera::setAspectRatio(float ar) {
    aspect_ratio = ar;
    computeProjectionMatrix();
}

void Camera::setSpeed(float newSpeed) { speed = newSpeed; }

void Camera::rotate(float yawDelta, float pitchDelta, float minPitch, float maxPitch) {
    yaw += yawDelta;
    pitch = glm::clamp(pitch + pitchDelta, minPitch, maxPitch);
}

void Camera::translate(const glm::vec3& delta) { position += delta * speed; }

// ----- TrackballCamera -----

TrackballCamera::TrackballCamera(const Camera& fromCamera, const glm::vec3& target)
    : Camera(fromCamera),
      target(target),
      lookAt(glm::normalize(target - fromCamera.getPosition())),
      radius(glm::length(target - fromCamera.getPosition())) {
    if (radius > 1e-6f) {
        glm::vec3 offsetDir = (fromCamera.getPosition() - target) / radius;
        pitch = glm::asin(offsetDir.z);
        yaw = glm::atan(offsetDir.y, offsetDir.x);
    }
}

void TrackballCamera::translate(const glm::vec3& delta) {
    glm::vec3 scaledDelta = delta * speed;
    position += scaledDelta;
    target += scaledDelta;
}

void TrackballCamera::zoom(float delta) { radius = glm::max(radius - delta * speed, 0.1f); }

glm::vec3 TrackballCamera::computeOrbitPosition() const {
    return target + radius * glm::vec3(glm::cos(pitch) * glm::cos(yaw), glm::cos(pitch) * glm::sin(yaw), glm::sin(pitch));
}

glm::vec3 TrackballCamera::getPosition() const { return computeOrbitPosition(); }

glm::vec3 TrackballCamera::getForward() const { return glm::normalize(target - computeOrbitPosition()); }

glm::vec3 TrackballCamera::getUp() const { return glm::cross(getRight(), getForward()); }

glm::mat4 TrackballCamera::getViewMatrix() const { return glm::lookAt(computeOrbitPosition(), target, up); }

// ----- FlyCamera -----

FlyCamera::FlyCamera(const Camera& fromCamera) : Camera(fromCamera) {
    position = fromCamera.getPosition();

    glm::vec3 forward = fromCamera.getForward();
    pitch = glm::asin(forward.z);
    yaw = glm::atan(forward.y, forward.x);
}

glm::vec3 FlyCamera::getForward() const {
    return glm::normalize(glm::vec3(glm::cos(pitch) * glm::cos(yaw), glm::cos(pitch) * glm::sin(yaw), glm::sin(pitch)));
}

void FlyCamera::moveForward(float dt) { translate(getForward() * dt); }

void FlyCamera::moveBackward(float dt) { translate(-getForward() * dt); }

void FlyCamera::moveRight(float dt) { translate(getRight() * dt); }

void FlyCamera::moveLeft(float dt) { translate(-getRight() * dt); }

void FlyCamera::moveUp(float dt) { translate(up * dt); }

void FlyCamera::moveDown(float dt) { translate(-up * dt); }

glm::mat4 FlyCamera::getViewMatrix() const { return glm::lookAt(position, position + getForward(), up); }
