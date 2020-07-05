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

void energy::Wrist::init(Camera *camera, SkeletonSerializer *skeleton, HandFinder *handfinder)
{
    this->camera = camera;
    this->skeleton = skeleton;
    this->handfinder = handfinder;

    /// TODO: AntTweakBar elements for wrist energy
}

void energy::Wrist::track(LinearSystem &system)
{

    if (!handfinder->wrist_found())
        return;
    if (!classifier_enable)
        return;

    /// @brief ugly hack to flip the direction of the PCA axis
    /// Ugly, but sufficient to get the teaser video recording!
    if (classifier_temporal)
    {
        static Vector3 prev_wrist_dir(0, 1, 0);
        if (handfinder->wrist_direction().dot(prev_wrist_dir) < 0)
            handfinder->wrist_direction_flip();
        prev_wrist_dir = handfinder->wrist_direction();
    }

    int hand_id = skeleton->getID("Hand");
    Vector3 hand_root = skeleton->getJoint("Hand")->getGlobalTranslation();
    Vector3 wrist_offpoint = handfinder->wrist_center() + handfinder->wrist_direction() * 100;

    Vector2 root_scr = camera->world_to_image(hand_root);
    Vector2 wrist_center_scr = camera->world_to_image(handfinder->wrist_center());
    Vector2 wrist_offpnt_scr = camera->world_to_image(wrist_offpoint);

    Vector2 n_wrist2 = (wrist_center_scr - wrist_offpnt_scr).normalized();
    n_wrist2 = Vector2(n_wrist2[1], -n_wrist2[0]);

    ///--- LHS
    Matrix_3xN J_sk = skeleton->jacobian(hand_id, hand_root);
    Matrix_2x3 J_pr = camera->projection_jacobian(hand_root);
    Matrix_1xN J = n_wrist2.transpose() * J_pr * J_sk;

    ///--- RHS
    Scalar rhs = n_wrist2.transpose() * (wrist_center_scr - root_scr);

    ///--- Add to solver
    Scalar weight = classifier_weight;
    system.lhs += weight * J.transpose() * J;
    system.rhs += weight * J.transpose() * rhs;

    // std::ofstream("lhs.txt") << transp(J) * J;
    // std::ofstream("rhs.txt") << transp(J) * rh
}

