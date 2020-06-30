#include "Wrist.h"

#include "util/gl_wrapper.h"
#include "util/opencv_wrapper.h"

#include "tracker/Data/Camera.h"
#include "tracker/HandFinder/HandFinder.h"
#include "tracker/DataStructure/SkeletonSerializer.h"
#include "tracker/OpenGL/DebugRenderer/DebugRenderer.h"


#ifdef DEBUG_VIZ
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
cv::Mat image; ///< debug
#endif

void energy::Wristband::init(Camera *camera, SkeletonSerializer *skeleton, HandFinder *handfinder)
{
    this->camera = camera;
    this->skeleton = skeleton;
    this->handfinder = handfinder;

    /// TODO: AntTweakBar elements for wrist energy
}

void energy::Wristband::track(LinearSystem &system)
{

    if (!handfinder->wristband_found())
        return;
    if (!classifier_enable)
        return;

    /// @brief ugly hack to flip the direction of the PCA axis
    if (classifier_temporal)
    {
        static Vector3 prev_wband_dir(0, 1, 0);
        if (handfinder->wristband_direction().dot(prev_wband_dir) < 0)
            handfinder->wristband_direction_flip();
        prev_wband_dir = handfinder->wristband_direction();
    }

    int hand_id = skeleton->getID("Hand");
    Vector3 hand_root = skeleton->getJoint("Hand")->getGlobalTranslation();
    Vector3 wband_offpoint = handfinder->wristband_center() + handfinder->wristband_direction() * 100;

    Vector2 root_scr = camera->world_to_image(hand_root);
    Vector2 wband_center_scr = camera->world_to_image(handfinder->wristband_center());
    Vector2 wband_offpnt_scr = camera->world_to_image(wband_offpoint);

#ifdef DEBUG_VIZ
    image = current_frame.color.clone();
    draw_circle(image, root_scr, cv::Scalar(255, 0, 0));
    draw_circle(image, wband_center_scr, cv::Scalar(0, 255, 0));
    draw_line(image, wband_center_scr, wband_offpnt_scr);
    cv::imshow("image", image);
#endif

    Vector2 n_wrist2 = (wband_center_scr - wband_offpnt_scr).normalized();
    n_wrist2 = Vector2(n_wrist2[1], -n_wrist2[0]);

    ///--- LHS
    Matrix_3xN J_sk = skeleton->jacobian(hand_id, hand_root);
    Matrix_2x3 J_pr = camera->projection_jacobian(hand_root);
    Matrix_1xN J = n_wrist2.transpose() * J_pr * J_sk;

    ///--- RHS
    Scalar rhs = n_wrist2.transpose() * (wband_center_scr - root_scr);

    ///--- Add to solver
    Scalar weight = classifier_weight;
    system.lhs += weight * J.transpose() * J;
    system.rhs += weight * J.transpose() * rhs;



}

