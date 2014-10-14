#include "articulate_HandModel_XYZRGB.h"
#include <opencv2/calib3d/calib3d.hpp>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <omp.h>
#include <vector>

#define PI 3.14159265

using namespace cv;


////////////////////////////////////////////////////////////
//*************      usefull functions     ***************//
Mat R_x(float theta){
    Mat Rx = Mat::zeros(3,3,CV_32FC1);
    Rx.at<float>(0,0) = 1;
    Rx.at<float>(1,1) = cos(theta*PI/180.0);
    Rx.at<float>(1,2) = -sin(theta*PI/180.0);
    Rx.at<float>(2,1) = sin(theta*PI/180.0);
    Rx.at<float>(2,2) = cos(theta*PI/180.0);
    return Rx;
}

Mat R_y(float theta){
    Mat Ry = Mat::zeros(3,3,CV_32FC1);
    Ry.at<float>(0,0) = cos(theta*PI/180.0);
    Ry.at<float>(0,2) = sin(theta*PI/180.0);
    Ry.at<float>(1,1) = 1;
    Ry.at<float>(2,0) = -sin(theta*PI/180.0);
    Ry.at<float>(2,2) = cos(theta*PI/180.0);
    return Ry;
}

Mat R_z(float theta){
    Mat Rz = Mat::zeros(3,3,CV_32FC1);
    Rz.at<float>(0,0) = cos(theta*PI/180.0);
    Rz.at<float>(0,1) = -sin(theta*PI/180.0);
    Rz.at<float>(1,0) = sin(theta*PI/180.0);
    Rz.at<float>(1,1) = cos(theta*PI/180.0);
    Rz.at<float>(2,2) = 1;
    return Rz;
}

double Distance(pcl::PointXYZRGB p1, pcl::PointXYZRGB p2){
    double dis = sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y)+(p1.z-p2.z)*(p1.z-p2.z));
    if (dis == 0)
        return 0.00000000001;
    else
        return dis;
}

double Length(Point3d p){
    double length = sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
    if (length == 0)
        return 0.00000000001;
    else
        return length;
}

//float Distance(pcl::PointXYZ p1, Mat p2){
//    float dis = sqrt((p1.x-p2.at<float>(0,0))*(p1.x-p2.at<float>(0,0))+(p1.y-p2.at<float>(1,0))*(p1.y-p2.at<float>(1,0))+(p1.z-p2.at<float>(2,0))*(p1.z-p2.at<float>(2,0)));
//    if (dis == 0)
//        return 0.00000001;
//    else
//        return dis;
//}

//float Distance(pcl::PointXYZ p1, pcl::PointXYZRGB p2){
//    float dis = sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y)+(p1.z-p2.z)*(p1.z-p2.z));
//    if (dis == 0)
//        return 0.00000001;
//    else
//        return dis;
//}

//float Distance(pcl::PointXYZ p1, pcl::PointXYZ p2){
//    float dis = sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y)+(p1.z-p2.z)*(p1.z-p2.z));
//    if (dis == 0)
//        return 0.00000001;
//    else
//        return dis;
//}

//float Distance(Point3d p1, Point3d p2){
//    float dis = sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y)+(p1.z-p2.z)*(p1.z-p2.z));
//    if (dis == 0)
//        return 0.00000001;
//    else
//        return dis;
//}

//float Distance(pcl::PointXYZ p1, Point3d p2){
//    float dis = sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y)+(p1.z-p2.z)*(p1.z-p2.z));
//    if (dis == 0)
//        return 0.00000001;
//    else
//        return dis;
//}

float arc2degree(float arc){
    return 180.0*arc/PI;
}

float degree2arc(float degree){
    return degree/180.0*PI;
}

/////////////////////////////////////////////////////////////////

articulate_HandModel_XYZRGB::articulate_HandModel_XYZRGB(const float finger_angle_step, const float xy_relosution)
{
    //1. joints color initialization:
    joints_position[0].rgb =  ((uint32_t)255 << 16 | (uint32_t)255 << 8 | (uint32_t)255);

    for(int finger_index = 0; finger_index < 5; finger_index ++){
        uint8_t rf = 63*finger_index;
        uint8_t bf = 255-rf;
        for(int j = 0; j<5;j++){
            uint8_t gf = j*50;
            uint32_t rgbf = ((uint32_t)rf << 16 | (uint32_t)gf << 8 | (uint32_t)bf);
            joints_position[finger_index*5+j+1].rgb = *reinterpret_cast<float*>(&rgbf);
            joints_position[finger_index*5+j+1].x = (finger_index-2)/15.0*(j*0.3+1);
            joints_position[finger_index*5+j+1].y = (j-2)/15.0;
            joints_position[finger_index*5+j+1].z = 0.0;
        }
    }
    joints_position[5] = joints_position[4];

    //2. bone length initialization:
    //thumb
    bone_length[0][0] = 0/1000;
    bone_length[0][1] = 40.1779/1000 - 0.002;
    bone_length[0][2] = 31.9564/1000;
    bone_length[0][3] = 22.9945/1000;
    //index finger
    bone_length[1][0] = 78.4271/1000;
    bone_length[1][1] = 46.0471/1000;
    bone_length[1][2] = 26.7806/1000;
    bone_length[1][3] = 19.517/1000;
    //middle finger
    bone_length[2][0] = 74.5294/1000;
    bone_length[2][1] = 50.4173/1000;
    bone_length[2][2] = 32.1543/1000;
    bone_length[2][3] = 22.2665/1000;
    //ring finger
    bone_length[3][0] = 67.2215/1000;
    bone_length[3][1] = 46.8076/1000;
    bone_length[3][2] = 31.4014/1000;
    bone_length[3][3] = 22.1557/1000;
    //pinky finger
    bone_length[4][0] = 62.4492/1000;
    bone_length[4][1] = 31.2519/1000;
    bone_length[4][2] = 21.0526/1000;
    bone_length[4][3] = 18.672/1000;

    //3. Model joints position initialization
    for(int i = 0; i < 26; i++){
        Model_joints[i] = Mat::zeros(3,1,CV_32FC1);
    }
    //3.1. palm joints: 1,6,11,16,21,7,12,17,22
    //palm joints with reference to palm/hand coordinate:
    //palm.thumb
    float delt_y = 0.002;

    Model_joints[1].at<float>(0,0) = -0.014 - 0.002;
    Model_joints[1].at<float>(1,0) = -0.053 +0.01+delt_y + 0.005;
    Model_joints[1].at<float>(2,0) = 0.002;
    //palm.index
    Model_joints[6].at<float>(0,0) = -0.014 - 0.002;
    Model_joints[6].at<float>(1,0) = -0.053 + delt_y;
    Model_joints[6].at<float>(2,0) = -0.008;

    Model_joints[7].at<float>(0,0) = -0.024 - 0.002;
    Model_joints[7].at<float>(1,0) = 0.019 + delt_y;
    Model_joints[7].at<float>(2,0) = 0;
    //palm.middle
    Model_joints[11].at<float>(0,0) = 0 - 0.002;
    Model_joints[11].at<float>(1,0) = -0.05 + delt_y;
    Model_joints[11].at<float>(2,0) = -0.008;

    Model_joints[12].at<float>(0,0) = -0.002 - 0.002;
    Model_joints[12].at<float>(1,0) = 0.023 + delt_y;
    Model_joints[12].at<float>(2,0) = -0.001;
    //palm.ring
    Model_joints[16].at<float>(0,0) = 0.014 - 0.002;
    Model_joints[16].at<float>(1,0) = -0.051 + delt_y;
    Model_joints[16].at<float>(2,0) = -0.008;

    Model_joints[17].at<float>(0,0) = 0.019 - 0.002;
    Model_joints[17].at<float>(1,0) = 0.018 + delt_y;
    Model_joints[17].at<float>(2,0) = 0.001;
    //palm.pinky
    Model_joints[21].at<float>(0,0) = 0.027 - 0.002;
    Model_joints[21].at<float>(1,0) = -0.053 + delt_y;
    Model_joints[21].at<float>(2,0) = -0.004;

    Model_joints[22].at<float>(0,0) = 0.038 - 0.002;
    Model_joints[22].at<float>(1,0) = 0.014 + delt_y;
    Model_joints[22].at<float>(2,0) = 0;

    for(int i = 0; i < 10; i++){
        auxiliary_palm_position[i] = Mat::zeros(3,1,CV_32FC1);
    }

    auxiliary_palm_position[0].at<float>(0,0) = 0;
    auxiliary_palm_position[0].at<float>(1,0) = 0;
    auxiliary_palm_position[0].at<float>(2,0) = 0;

    auxiliary_palm_position[1].at<float>(0,0) = 1;
    auxiliary_palm_position[1].at<float>(1,0) = 0;
    auxiliary_palm_position[1].at<float>(2,0) = 0;

    auxiliary_palm_position[2].at<float>(0,0) = 0;
    auxiliary_palm_position[2].at<float>(1,0) = 1;
    auxiliary_palm_position[2].at<float>(2,0) = 0;

    auxiliary_palm_position[3].at<float>(0,0) = 0;
    auxiliary_palm_position[3].at<float>(1,0) = 0;
    auxiliary_palm_position[3].at<float>(2,0) = 1;

    auxiliary_palm_position[4].at<float>(0,0) = Model_joints[21].at<float>(0,0);
    auxiliary_palm_position[4].at<float>(1,0) = Model_joints[21].at<float>(1,0);
    auxiliary_palm_position[4].at<float>(2,0) = Model_joints[21].at<float>(2,0) - 0.01;

    auxiliary_palm_position[5].at<float>(0,0) = Model_joints[22].at<float>(0,0);
    auxiliary_palm_position[5].at<float>(1,0) = Model_joints[22].at<float>(1,0);
    auxiliary_palm_position[5].at<float>(2,0) = Model_joints[22].at<float>(2,0) - 0.01;

    auxiliary_palm_position[6].at<float>(0,0) = Model_joints[6].at<float>(0,0);
    auxiliary_palm_position[6].at<float>(1,0) = Model_joints[6].at<float>(1,0);
    auxiliary_palm_position[6].at<float>(2,0) = Model_joints[6].at<float>(2,0) + 0.01;

    auxiliary_palm_position[7].at<float>(0,0) = Model_joints[7].at<float>(0,0);
    auxiliary_palm_position[7].at<float>(1,0) = Model_joints[7].at<float>(1,0);
    auxiliary_palm_position[7].at<float>(2,0) = Model_joints[7].at<float>(2,0) + 0.01;

    auxiliary_palm_position[8].at<float>(0,0) = Model_joints[21].at<float>(0,0);
    auxiliary_palm_position[8].at<float>(1,0) = Model_joints[21].at<float>(1,0);
    auxiliary_palm_position[8].at<float>(2,0) = Model_joints[21].at<float>(2,0) + 0.01;

    auxiliary_palm_position[9].at<float>(0,0) = Model_joints[22].at<float>(0,0);
    auxiliary_palm_position[9].at<float>(1,0) = Model_joints[22].at<float>(1,0);
    auxiliary_palm_position[9].at<float>(2,0) = Model_joints[22].at<float>(2,0) + 0.01;


    for(int i = 0; i< 4; i++){
        auxiliary_palm_position_now[i] = Mat::zeros(3,1,CV_32FC1);
    }
    //3.2.fingers:
    //3.2.1 index(extrinsic):
    Model_joints[8].at<float>(1,0) = bone_length[1][1];
    Model_joints[9].at<float>(1,0) = bone_length[1][2];
    Model_joints[10].at<float>(1,0) = bone_length[1][3];

    //3.2.2 middel to pinky(extrinsic):
    for ( int i = 0; i < 3; ++i){
        Model_joints[i*5+13].at<float>(1,0) = bone_length[2+i][1];
        Model_joints[i*5+14].at<float>(1,0) = bone_length[2+i][2];
        Model_joints[i*5+15].at<float>(1,0) = bone_length[2+i][3];

    }

    //3.2.3 thumb(extrinsic)
    Model_joints[2].at<float>(1,0) = bone_length[0][1];
    Model_joints[3].at<float>(1,0) = bone_length[0][2];
    Model_joints[4].at<float>(1,0) = bone_length[0][3];

    palm_model = Mat::zeros(3, 9, CV_32FC1);

    Model_joints[6].copyTo(palm_model.col(0));
    Model_joints[7].copyTo(palm_model.col(1));
    Model_joints[11].copyTo(palm_model.col(2));
    Model_joints[12].copyTo(palm_model.col(3));
    Model_joints[16].copyTo(palm_model.col(4));
    Model_joints[17].copyTo(palm_model.col(5));
    Model_joints[21].copyTo(palm_model.col(6));
    Model_joints[22].copyTo(palm_model.col(7));
    Model_joints[1].copyTo(palm_model.col(8));

    set_parameters();

    //5. parameter min and max:
    parameters_min[0] = -10;
    parameters_max[0] = 10;
    parameters_min[1] = -10;
    parameters_max[1] = 10;
    parameters_min[2] = -10;
    parameters_max[2] = 10;

    parameters_min[3] = -180;
    parameters_max[3] = 180;
    parameters_min[4] = -180;
    parameters_max[4] = 180;
    parameters_min[5] = -180;
    parameters_max[5] = 180;

    parameters_min[6] = 5;
    parameters_max[6] = 50;
    parameters_min[7] = -60;
    parameters_max[7] = -10;
    parameters_min[8] = -10;
    parameters_max[8] = 70;
    parameters_min[9] = -10;
    parameters_max[9] = 90;

    parameters_min[10] = -20;
    parameters_max[10] = 30;
    parameters_min[11] = -10;
    parameters_max[11] = 30;
    parameters_min[12] = -5;
    parameters_max[12] = 110;
    parameters_min[13] = 0;
    parameters_max[13] = 80;

    parameters_min[14] = -20;
    parameters_max[14] = 20;
    parameters_min[15] = -10;
    parameters_max[15] = 30;
    parameters_min[16] = -5;
    parameters_max[16] = 110;
    parameters_min[17] = 0;
    parameters_max[17] = 80;

    parameters_min[18] = -20;
    parameters_max[18] = 5;
    parameters_min[19] = -10;
    parameters_max[19] = 30;
    parameters_min[20] = -5;
    parameters_max[20] = 110;
    parameters_min[21] = 0;
    parameters_max[21] = 80;

    parameters_min[22] = -30;
    parameters_max[22] = 0;
    parameters_min[23] = -20;
    parameters_max[23] = 30;
    parameters_min[24] = -5;
    parameters_max[24] = 110;
    parameters_min[25] = 0;
    parameters_max[25] = 80;

    parameters_min[26] = 60;
    parameters_max[26] = 90;


    //std::cout << "Model is ready!" << std::endl;

}

bool articulate_HandModel_XYZRGB::check_parameters(int &wrong_parameter_index){
    for(int i = 0; i < 27; i++){
        if(parameters[i] > parameters_max[i] || parameters[i] < parameters_min[i]){
            wrong_parameter_index = i;
            std::cout << "Wrong parameter index: " << wrong_parameter_index <<"; Value: "<< parameters[i] << std::endl;
            return false;
        }
    }
    return true;
}

void articulate_HandModel_XYZRGB::check_parameters(){
    for(int i = 0; i < 27; i++){
        if(parameters[i] > parameters_max[i]){
            parameters[i] = parameters_max[i];
        }
        else if (parameters[i] < parameters_min[i]){
            parameters[i] = parameters_min[i];
        }
    }
}

void articulate_HandModel_XYZRGB::set_parameters(){
    //0: hand center position x;
    //1: hand center position y;
    //2: hand center position z;
    parameters[0] = 0;
    parameters[1] = 0;
    parameters[2] = 0;
    //3: palm rotation angle around x axis;
    //4: palm rotation angle around y axis;
    //5: palm rotation angle around z axis;
    parameters[3] = -30;
    parameters[4] = 0;
    parameters[5] = 0;
    //6: horizontal angle between thumb metacarpal and palm;
    //7: vertical angle between thumb metacarpal and palm;
    //8: angle between thumb metacarpal and proximal;
    //9: angle between thumb proximal and distal;
    parameters[6] = 20;
    parameters[7] = 20;
    parameters[8] = 30;
    parameters[9] = 10;
    //10: horizontal angle between index finger proximal and palm;
    //11: vertical angle between index finger proximal and palm;
    //12: angle between index finger proximal and intermediate;
    //13: angle between index finger intermediate and distal;
    parameters[10] = 10;
    parameters[11] = 0;
    parameters[12] = 0;
    parameters[13] = 0;
    //14: horizontal angle between middle finger proximal and palm;
    //15: vertical angle between middle finger proximal and palm;
    //16: angle between middle finger proximal and intermediate;
    //17: angle between middle finger intermediate and distal;
    parameters[14] = 0;
    parameters[15] = 0;
    parameters[16] = 0;
    parameters[17] = 1;
    //18: horizontal angle between ring finger proximal and palm;
    //19: vertical angle between ring finger proximal and palm;
    //20: angle between ring finger proximal and intermediate;
    //21: angle between ring finger intermediate and distal;
    parameters[18] = -10;
    parameters[19] = 39;
    parameters[20] = 39;
    parameters[21] = 45;
    //22: horizontal angle between pinky proximal and palm;
    //23: vertical angle between pinky proximal and palm;
    //24: angle between pinky proximal and intermediate;
    //25: angle between pinky intermediate and distal;
    parameters[22] = -25;
    parameters[23] = 30;
    parameters[24] = 70;
    parameters[25] = 30;

    parameters[26] = 70;
}

void articulate_HandModel_XYZRGB::set_parameters(float para[27]){
    for(int i = 0; i<27; i++)
        parameters[i] = para[i];
}

void articulate_HandModel_XYZRGB::set_parameters(vector<float> para){
    if(para.size() == 27){
        for(int i = 0; i<27; i++)
            parameters[i] = para[i];
    }
    else{
        ROS_ERROR("Parameter size error!");
    }
}

void articulate_HandModel_XYZRGB::get_joints_positions(){

    Mat joints_for_calc[26];
    for(int i = 0; i < 26; i++){
        joints_for_calc[i] = Mat::zeros(3,1,CV_32FC1);
        Model_joints[i].copyTo(joints_for_calc[i]);
    }
    //1. palm joints: 1,6,11,16,21,7,12,17,22
    //palm joints with reference to palm/hand coordinate:
    //palm.thumb
    Model_joints[1].copyTo(joints_for_calc[1]);
    //palm.index
    Model_joints[6].copyTo(joints_for_calc[6]);
    Model_joints[7].copyTo(joints_for_calc[7]);
    //palm.middle
    Model_joints[11].copyTo(joints_for_calc[11]);
    Model_joints[12].copyTo(joints_for_calc[12]);
    //palm.ring
    Model_joints[16].copyTo(joints_for_calc[16]);
    Model_joints[17].copyTo(joints_for_calc[17]);
    //palm.pinky
    Model_joints[21].copyTo(joints_for_calc[21]);
    Model_joints[22].copyTo(joints_for_calc[22]);

    //2.fingers:
    //2.1 index(intrinsic, from left to right):
    Mat R[3];
    R[0] = R_z(parameters[10])*R_x(parameters[11]);
    R[1] = R_x(parameters[12]);
    R[2] = R_x(parameters[13]);

    joints_for_calc[10] = R[0]*(R[1]*(R[2]*joints_for_calc[10]+joints_for_calc[9])+joints_for_calc[8])+joints_for_calc[7];
    joints_for_calc[9] = R[0]*(R[1]*joints_for_calc[9]+joints_for_calc[8])+joints_for_calc[7];
    joints_for_calc[8] = R[0]*joints_for_calc[8]+joints_for_calc[7];

    //2.2 middel to pinky(intrinsic, from left to right):
    for ( int i = 0; i < 3; ++i){
        R[0] = R_z(parameters[i*4+14])*R_x(parameters[i*4+15]);
        R[1] = R_x(parameters[i*4+16]);
        R[2] = R_x(parameters[i*4+17]);

        joints_for_calc[i*5+15] = R[0]*(R[1]*(R[2]*joints_for_calc[i*5+15]+joints_for_calc[i*5+14])+joints_for_calc[i*5+13])+joints_for_calc[i*5+12];
        joints_for_calc[i*5+14] = R[0]*(R[1]*joints_for_calc[i*5+14]+joints_for_calc[i*5+13])+joints_for_calc[i*5+12];
        joints_for_calc[i*5+13] = R[0]*joints_for_calc[i*5+13]+joints_for_calc[i*5+12];

    }

    //2.3 thumb(intrinsic, from left to right):
    //R[0] = R_y(90);
    R[0] = R_y(parameters[26])*R_z(parameters[6])*R_x(parameters[7]);
    R[1] = R_x(parameters[8]);
    R[2] = R_x(parameters[9]);

//    R[0] = R_y(parameters[6])*R_x(parameters[7]);
//    R[1] = R_x(parameters[8]);
//    R[2] = R_x(parameters[9]);

    //    cv::Mat mtxR, mtxQ;
    //    cv::Vec3d angles;
    //    angles = cv::RQDecomp3x3(R[0]*R_y(70).inv(), mtxR, mtxQ);
    //    std::cout<<"0: "<<angles[0]<<std::endl;
    //    std::cout<<"1: "<<angles[1]<<std::endl;
    //    std::cout<<"2: "<<angles[2]<<std::endl;

    joints_for_calc[4] = R[0]*(R[1]*(R[2]*joints_for_calc[4]+joints_for_calc[3])+joints_for_calc[2])+joints_for_calc[1];
    joints_for_calc[3] = R[0]*(R[1]*joints_for_calc[3]+joints_for_calc[2])+joints_for_calc[1];
    joints_for_calc[2] = R[0]*joints_for_calc[2]+joints_for_calc[1];
    joints_for_calc[5].at<float>(0,0) = joints_for_calc[4].at<float>(0,0);
    joints_for_calc[5].at<float>(1,0) = joints_for_calc[4].at<float>(1,0);
    joints_for_calc[5].at<float>(2,0) = joints_for_calc[4].at<float>(2,0);


    //3. palm after pitch yaw roll(extrinsic):
    Mat R_p_r_y = R_z(parameters[5])*R_y(parameters[4])*R_x(parameters[3]);
    //std::cout << "R: " << R_p_r_y << std::endl;
    Mat translation = Mat::zeros(3,1,CV_32FC1);
    translation.at<float>(0,0) = parameters[0];
    translation.at<float>(1,0) = parameters[1];
    translation.at<float>(2,0) = parameters[2];

    //std::cout << "translation: " << translation << std::endl;
    for(int i = 0; i< 26; i++){
        joints_for_calc[i] = R_p_r_y * joints_for_calc[i]+translation;
    }

    //3.1 palm auxiliary points:
    for(int i = 0; i< 4; i++){
        auxiliary_palm_position[i].copyTo(auxiliary_palm_position_now[i]);
        auxiliary_palm_position_now[i] = R_p_r_y * auxiliary_palm_position_now[i]+translation;
    }


    //4. put calculation results into joints_position
    for(int i = 0; i< 26; i++){
        joints_position[i].x = joints_for_calc[i].at<float>(0,0);
        joints_position[i].y = joints_for_calc[i].at<float>(0,1);
        joints_position[i].z = joints_for_calc[i].at<float>(0,2);
        //std::cout<< i <<": "<<joints_position[i]<<std::endl;
    }

}

void articulate_HandModel_XYZRGB::get_handPointCloud(pcl::PointCloud<pcl::PointXYZRGB> & handPointCloud){
    handPointCloud.clear();
    Mat R_p_r_y = R_z(parameters[5])*R_y(parameters[4])*R_x(parameters[3]);
    //std::cout << "R: " << R_p_r_y << std::endl;
    Mat translation = Mat::zeros(3,1,CV_32FC1);
    translation.at<float>(0,0) = parameters[0];
    translation.at<float>(1,0) = parameters[1];
    translation.at<float>(2,0) = parameters[2];

    //index finger to pinky finger point cloud
    pcl::PointXYZRGB temp_p;

    for(int fingerIndex = 1; fingerIndex<5; ++fingerIndex){
        Mat R[3];
        R[0] = R_z(parameters[fingerIndex*4 + 6])*R_x(parameters[fingerIndex*4 + 7]);
        R[1] = R_x(parameters[fingerIndex*4 + 8]);
        R[2] = R_x(parameters[fingerIndex*4 + 9]);
        for( int boneIndex = 0; boneIndex < 3; ++boneIndex){

            int vectorIndex = fingerIndex*3+1+boneIndex;
            temp_p.rgb = handPointCloudVector[vectorIndex].points[0].rgb;

            //#pragma omp parallel for
            for(int i = 0; i < handPointCloudVector[vectorIndex].points.size(); ++i){
                Mat point4calcu = Mat::zeros(3,1,CV_32FC1);
                point4calcu.at<float>(0,0) = handPointCloudVector[vectorIndex].points[i].x;
                point4calcu.at<float>(1,0) = handPointCloudVector[vectorIndex].points[i].y;
                point4calcu.at<float>(2,0) = handPointCloudVector[vectorIndex].points[i].z;

                if(boneIndex == 0)
                    point4calcu = R_p_r_y * (R[0]*point4calcu+Model_joints[fingerIndex*5+2]) + translation;
                else if (boneIndex ==1){
                    point4calcu = R_p_r_y * (R[0]*(R[1]*point4calcu+Model_joints[fingerIndex*5+3])+Model_joints[fingerIndex*5+2]) + translation;
                }
                else{
                    point4calcu = R_p_r_y * (R[0]*(R[1]*(R[2]*point4calcu+Model_joints[fingerIndex*5+4])+Model_joints[fingerIndex*5+3])+Model_joints[fingerIndex*5+2])+ translation;
                }

                temp_p.x = point4calcu.at<float>(0,0);
                temp_p.y = point4calcu.at<float>(1,0);
                temp_p.z = point4calcu.at<float>(2,0);
                //                handPointCloudVector[vectorIndex].points[i].x = point4calcu.at<float>(0,0);
                //                handPointCloudVector[vectorIndex].points[i].y = point4calcu.at<float>(1,0);
                //                handPointCloudVector[vectorIndex].points[i].z = point4calcu.at<float>(2,0);

                handPointCloud.push_back(temp_p);
            }
        }
    }
    //thum point cloud:
    Mat R[3];
    R[0] = R_y(70);
    R[0] = R_z(10+parameters[6])*R_x(parameters[7])*R[0];
    R[1] = R_x(parameters[8]);
    R[2] = R_x(parameters[9]);
    for( int boneIndex = 0; boneIndex < 3; ++boneIndex){
        int vectorIndex = boneIndex+1;
        temp_p.rgb = handPointCloudVector[vectorIndex].points[0].rgb;

#pragma omp parallel for
        for(int i = 0; i < handPointCloudVector[vectorIndex].points.size(); ++i){
            Mat point4calcu = Mat::zeros(3,1,CV_32FC1);
            point4calcu.at<float>(0,0) = handPointCloudVector[vectorIndex].points[i].x;
            point4calcu.at<float>(1,0) = handPointCloudVector[vectorIndex].points[i].y;
            point4calcu.at<float>(2,0) = handPointCloudVector[vectorIndex].points[i].z;

            if(boneIndex == 0)
                point4calcu = R_p_r_y * (R[0]*point4calcu+Model_joints[1]) + translation;
            else if (boneIndex ==1){
                point4calcu = R_p_r_y * (R[0]*(R[1]*point4calcu+Model_joints[2])+Model_joints[1]) + translation;
            }
            else{
                point4calcu = R_p_r_y * (R[0]*(R[1]*(R[2]*point4calcu+Model_joints[3])+Model_joints[2])+Model_joints[1])+ translation;
            }

            temp_p.x = point4calcu.at<float>(0,0);
            temp_p.y = point4calcu.at<float>(1,0);
            temp_p.z = point4calcu.at<float>(2,0);
            //                handPointCloudVector[vectorIndex].points[i].x = point4calcu.at<float>(0,0);
            //                handPointCloudVector[vectorIndex].points[i].y = point4calcu.at<float>(1,0);
            //                handPointCloudVector[vectorIndex].points[i].z = point4calcu.at<float>(2,0);

            handPointCloud.push_back(temp_p);
        }
    }
    //palm point cloud:
    temp_p.rgb = handPointCloudVector[0].points[0].rgb;
#pragma omp parallel for
    for(int i = 0; i < handPointCloudVector[0].points.size(); ++i){
        Mat point4calcu = Mat::zeros(3,1,CV_32FC1);
        point4calcu.at<float>(0,0) = handPointCloudVector[0].points[i].x;
        point4calcu.at<float>(1,0) = handPointCloudVector[0].points[i].y;
        point4calcu.at<float>(2,0) = handPointCloudVector[0].points[i].z;

        point4calcu = R_p_r_y * point4calcu + translation;

        temp_p.x = point4calcu.at<float>(0,0);
        temp_p.y = point4calcu.at<float>(1,0);
        temp_p.z = point4calcu.at<float>(2,0);
        //                handPointCloudVector[vectorIndex].points[i].x = point4calcu.at<float>(0,0);
        //                handPointCloudVector[vectorIndex].points[i].y = point4calcu.at<float>(1,0);
        //                handPointCloudVector[vectorIndex].points[i].z = point4calcu.at<float>(2,0);

        handPointCloud.push_back(temp_p);
    }

    //    //joints ball point cloud:
    //    for(int i = 0; i< 26; ++i){
    //use uniform sphere to creat joint ball point cloud around joints:
    //        joints_position[i].x;
    //    }


}

void articulate_HandModel_XYZRGB::samplePointCloud(pcl::PointCloud<pcl::PointXYZRGB> & handPointCloud){
    handPointCloud.clear();

//    vector<double> foo;
//    foo.resize(200000);
//    int zhe = 200000;
//#pragma omp parallel for
//    for(int j = 0; j<4;j++){
//        for(int i = 0; i< zhe/4; i++){
//            float value;
//            value = i*3-j+64.0/i;
//            if(value >3)
//                value = 3;
//            else
//                value = 2;
//           // foo[5000*j+i] = value;
//        }
//    }

    float xy_resolution = 0.003;
    float theta_resolution = 15;
    //1. Fingers:
    //1.a Thumb:
    for(int bone_index = 0; bone_index < 3; bone_index++){
        int jointStart = bone_index+1;
        int jointEnd = bone_index+2;
        //1.1 axis of cylinder:
        Point3d axis_vector;
        double axis_vector_length = Distance(joints_position[jointEnd], joints_position[jointStart]);
        axis_vector.x = (joints_position[jointEnd].x - joints_position[jointStart].x)/axis_vector_length;
        axis_vector.y = (joints_position[jointEnd].y - joints_position[jointStart].y)/axis_vector_length;
        axis_vector.z = (joints_position[jointEnd].z - joints_position[jointStart].z)/axis_vector_length;

        //1.2 radius vectors:
        //1.2.1 first radius vector:
        Point3d radius_vector1;
        if(axis_vector.x > 0.01 || axis_vector.x < -0.01){
            radius_vector1.y = 1;
            radius_vector1.z = 1;
            radius_vector1.x = -(axis_vector.y + axis_vector.z)/axis_vector.x;
        }
        else if(abs(axis_vector.y || axis_vector.y < -0.01) > 0.01){
            radius_vector1.x = 1;
            radius_vector1.z = 1;
            radius_vector1.y = -(axis_vector.x + axis_vector.z)/axis_vector.y;
        }
        else if(abs(axis_vector.z || axis_vector.z < -0.01) > 0.01){
            radius_vector1.x = 1;
            radius_vector1.y = 1;
            radius_vector1.z = -(axis_vector.x + axis_vector.y)/axis_vector.z;
        }
        else{
            ROS_INFO("Error in 'articulate_HandModel_XYZRGB' radius vector calculation!");
        }

        double radius_vector1_length = Length(radius_vector1);
        radius_vector1.x /= radius_vector1_length;
        radius_vector1.y /= radius_vector1_length;
        radius_vector1.z /= radius_vector1_length;

        //std::cout<< " radius vector 1: " << radius_vector1 << std::endl;

        //1.2.2 second radius vector:
        Point3d radius_vector2;
        radius_vector2.x = radius_vector1.y*axis_vector.z - radius_vector1.z*axis_vector.y;
        radius_vector2.y = radius_vector1.z*axis_vector.x - radius_vector1.x*axis_vector.z;
        radius_vector2.z = radius_vector1.x*axis_vector.y - radius_vector1.y*axis_vector.x;

        //    std::cout<< " axis vector: " << axis_vector << std::endl;
        //    std::cout<< " radius vector 1: " << radius_vector1 << std::endl;
        //    std::cout<< " radius vector 2: " << radius_vector2 << std::endl;

        //1.3 sample cylinder:
        float radius = 0.015;

        pcl::PointXYZRGB p;
        p.rgb = joints_position[jointEnd].rgb;

        float step = 0;
        float maxi_step = bone_length[0][bone_index+1];
        if(bone_index == 2)
            maxi_step -= 0.007;
        while(step < maxi_step){
            Point3d axis_step;
            axis_step.x = step * axis_vector.x + joints_position[jointStart].x;
            axis_step.y = step * axis_vector.y + joints_position[jointStart].y;
            axis_step.z = step * axis_vector.z + joints_position[jointStart].z;

            if(bone_index == 0)
                radius = 0.01/*+0.001/bone_length[0][1]*step*/;
            else if (bone_index ==1)
                radius = 0.011 - 0.002/bone_length[0][2]*step;
            else
                radius = 0.009 - 0.001/bone_length[0][3]*step;

            for(int theta = 0; theta < 360; theta += theta_resolution){
                float cosT = cos(degree2arc(theta));
                float sinT = sin(degree2arc(theta));
                p.x = radius*(radius_vector1.x*cosT + radius_vector2.x*sinT) + axis_step.x;
                p.y = radius*(radius_vector1.y*cosT + radius_vector2.y*sinT) + axis_step.y;
                p.z = radius*(radius_vector1.z*cosT + radius_vector2.z*sinT) + axis_step.z;

                handPointCloud.push_back(p);

            }
            step += xy_resolution;
        }
        step -= xy_resolution;
        float phi = theta_resolution;
        while(phi < 90){
            Point3d axis_step;
            axis_step.x = (step + radius*sin(degree2arc(phi))) * axis_vector.x + joints_position[jointStart].x;
            axis_step.y = (step + radius*sin(degree2arc(phi))) * axis_vector.y + joints_position[jointStart].y;
            axis_step.z = (step + radius*sin(degree2arc(phi))) * axis_vector.z + joints_position[jointStart].z;
            for(int theta = 0; theta < 360; theta += phi ){
                float cosT = cos(degree2arc(theta));
                float sinT = sin(degree2arc(theta));
                float temp_radius = radius * cos(degree2arc(phi));
                p.x = temp_radius*(radius_vector1.x*cosT + radius_vector2.x*sinT) + axis_step.x;
                p.y = temp_radius*(radius_vector1.y*cosT + radius_vector2.y*sinT) + axis_step.y;
                p.z = temp_radius*(radius_vector1.z*cosT + radius_vector2.z*sinT) + axis_step.z;

                handPointCloud.push_back(p);

            }
            phi += theta_resolution;
        }
    }

    //Fingers from 2 to 5:
    for(int finger_index = 1; finger_index < 5; finger_index++ ){
        for(int bone_index = 0; bone_index < 3; bone_index++){
            int jointStart = 2+5*finger_index + bone_index;
            int jointEnd = jointStart + 1;
            //1.1 axis of cylinder:
            Point3d axis_vector;
            double axis_vector_length = Distance(joints_position[jointEnd], joints_position[jointStart]);
            axis_vector.x = (joints_position[jointEnd].x - joints_position[jointStart].x)/axis_vector_length;
            axis_vector.y = (joints_position[jointEnd].y - joints_position[jointStart].y)/axis_vector_length;
            axis_vector.z = (joints_position[jointEnd].z - joints_position[jointStart].z)/axis_vector_length;

            //1.2 radius vectors:
            //1.2.1 first radius vector:
            Point3d radius_vector1;
            if(axis_vector.x > 0.01 || axis_vector.x < -0.01){
                radius_vector1.y = 1;
                radius_vector1.z = 1;
                radius_vector1.x = -(axis_vector.y + axis_vector.z)/axis_vector.x;
            }
            else if(abs(axis_vector.y || axis_vector.y < -0.01) > 0.01){
                radius_vector1.x = 1;
                radius_vector1.z = 1;
                radius_vector1.y = -(axis_vector.x + axis_vector.z)/axis_vector.y;
            }
            else if(abs(axis_vector.z || axis_vector.z < -0.01) > 0.01){
                radius_vector1.x = 1;
                radius_vector1.y = 1;
                radius_vector1.z = -(axis_vector.x + axis_vector.y)/axis_vector.z;
            }
            else{
                ROS_INFO("Error in 'articulate_HandModel_XYZRGB' radius vector calculation!");
            }

            double radius_vector1_length = Length(radius_vector1);
            radius_vector1.x /= radius_vector1_length;
            radius_vector1.y /= radius_vector1_length;
            radius_vector1.z /= radius_vector1_length;

            //1.2.2 second radius vector:
            Point3d radius_vector2;
            radius_vector2.x = radius_vector1.y*axis_vector.z - radius_vector1.z*axis_vector.y;
            radius_vector2.y = radius_vector1.z*axis_vector.x - radius_vector1.x*axis_vector.z;
            radius_vector2.z = radius_vector1.x*axis_vector.y - radius_vector1.y*axis_vector.x;

            //    std::cout<< " axis vector: " << axis_vector << std::endl;
            //    std::cout<< " radius vector 1: " << radius_vector1 << std::endl;
            //    std::cout<< " radius vector 2: " << radius_vector2 << std::endl;

            //1.3 sample cylinder:
            float radius = 0.01;

            pcl::PointXYZRGB p;
            p.rgb = joints_position[jointEnd].rgb;

            float step = 0;
            float maxi_step = bone_length[finger_index][bone_index + 1];
            if(bone_index == 2){
                    if(finger_index !=4)
                     maxi_step -= 0.007;
                    else
                        maxi_step -= 0.006;
            }
            while(step < maxi_step){
                float theta = 0;
                Point3d axis_step;
                axis_step.x = step * axis_vector.x + joints_position[jointStart].x;
                axis_step.y = step * axis_vector.y + joints_position[jointStart].y;
                axis_step.z = step * axis_vector.z + joints_position[jointStart].z;

                radius = 0.01-(bone_index)*0.001-0.001/bone_length[finger_index][bone_index+1]*step;
                if(finger_index == 4)
                    radius -= 0.002;
                while(theta < 360){
                    float cosT = cos(degree2arc(theta));
                    float sinT = sin(degree2arc(theta));
                    p.x = radius*(radius_vector1.x*cosT + radius_vector2.x*sinT) + axis_step.x;
                    p.y = radius*(radius_vector1.y*cosT + radius_vector2.y*sinT) + axis_step.y;
                    p.z = radius*(radius_vector1.z*cosT + radius_vector2.z*sinT) + axis_step.z;

                    handPointCloud.push_back(p);

                    theta += theta_resolution;

                }
                step += xy_resolution;
            }
            step -= xy_resolution;
            float phi = theta_resolution;
            while(phi < 90){
                Point3d axis_step;
                float sinPhi = sin(degree2arc(phi));
                float cosPhi = cos(degree2arc(phi));
                axis_step.x = (step + radius*sinPhi) * axis_vector.x + joints_position[jointStart].x;
                axis_step.y = (step + radius*sinPhi) * axis_vector.y + joints_position[jointStart].y;
                axis_step.z = (step + radius*sinPhi) * axis_vector.z + joints_position[jointStart].z;
                for(int theta = 0; theta < 360; theta += phi ){
                    float cosT = cos(degree2arc(theta));
                    float sinT = sin(degree2arc(theta));
                    float temp_radius = radius * cosPhi;
                    p.x = temp_radius*(radius_vector1.x*cosT + radius_vector2.x*sinT) + axis_step.x;
                    p.y = temp_radius*(radius_vector1.y*cosT + radius_vector2.y*sinT) + axis_step.y;
                    p.z = temp_radius*(radius_vector1.z*cosT + radius_vector2.z*sinT) + axis_step.z;

                    handPointCloud.push_back(p);

                }
                phi += theta_resolution;
            }
        }
    }



    //2. Palm:
    Point3d vecX, vecY, vecZ;
    vecX.x = auxiliary_palm_position_now[1].at<float>(0,0) - auxiliary_palm_position_now[0].at<float>(0,0);
    vecX.y = auxiliary_palm_position_now[1].at<float>(1,0) - auxiliary_palm_position_now[0].at<float>(1,0);
    vecX.z = auxiliary_palm_position_now[1].at<float>(2,0) - auxiliary_palm_position_now[0].at<float>(2,0);

    vecY.x = auxiliary_palm_position_now[2].at<float>(0,0) - auxiliary_palm_position_now[0].at<float>(0,0);
    vecY.y = auxiliary_palm_position_now[2].at<float>(1,0) - auxiliary_palm_position_now[0].at<float>(1,0);
    vecY.z = auxiliary_palm_position_now[2].at<float>(2,0) - auxiliary_palm_position_now[0].at<float>(2,0);

    vecZ.x = auxiliary_palm_position_now[3].at<float>(0,0) - auxiliary_palm_position_now[0].at<float>(0,0);
    vecZ.y = auxiliary_palm_position_now[3].at<float>(1,0) - auxiliary_palm_position_now[0].at<float>(1,0);
    vecZ.z = auxiliary_palm_position_now[3].at<float>(2,0) - auxiliary_palm_position_now[0].at<float>(2,0);

    //2.1 palm front:
    pcl::PointXYZRGB p;
    p.rgb = joints_position[0].rgb;
    for(float step = -0.036; step < 0.049; step += xy_resolution){
        Point3d center;
        center.x = auxiliary_palm_position_now[0].at<float>(0,0) + step*vecX.x + 0.012*vecY.x;
        center.y = auxiliary_palm_position_now[0].at<float>(1,0) + step*vecX.y + 0.012*vecY.y;
        center.z = auxiliary_palm_position_now[0].at<float>(2,0) + step*vecX.z + 0.012*vecY.z;

        float radius = 0.012 - fabs(step - 0.0095)*0.1;
        //float radius_factor = 1-fabs(step - 0.0095)*0.1;
        for(int theta = theta_resolution; theta < 180; theta += theta_resolution){
            float cosT = cos(degree2arc(theta)), sinT = sin(degree2arc(theta));

            p.x = center.x + radius*(vecY.x*sinT + vecZ.x*cosT);
            p.y = center.y + radius*(vecY.y*sinT + vecZ.y*cosT);
            p.z = center.z + radius*(vecY.z*sinT + vecZ.z*cosT);

            handPointCloud.push_back(p);
        }
    }

    //2.2 palm main:
    uint8_t r = 50;
    uint8_t g = 255;
    uint8_t b = 50;
    uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
    p.rgb = rgb;
    Point3d start;
    start.x = auxiliary_palm_position_now[0].at<float>(0,0) + 0.012*vecY.x + 0.005*vecX.x;
    start.y = auxiliary_palm_position_now[0].at<float>(1,0) + 0.012*vecY.y + 0.005*vecX.y;
    start.z = auxiliary_palm_position_now[0].at<float>(2,0) + 0.012*vecY.z + 0.005*vecX.z;

    for(float step = 0; step < 0.075; step += xy_resolution){
        Point3d center;
        center.x = start.x - step * vecY.x;
        center.y = start.y - step * vecY.y;
        center.z = start.z - step * vecY.z;

        float radius = 0.01;
        float radius_factor1 = (8.5 - step * 40)/2.0;
        float radius_factor2 = 1.2 + step * 5;
        for(float theta = 0; theta < 360; theta += 5){
            float cosT = cos(degree2arc(theta)), sinT = sin(degree2arc(theta));

            p.x = center.x + radius*(radius_factor1*vecX.x*sinT + radius_factor2*vecZ.x*cosT);
            p.y = center.y + radius*(radius_factor1*vecX.y*sinT + radius_factor2*vecZ.y*cosT);
            p.z = center.z + radius*(radius_factor1*vecX.z*sinT + radius_factor2*vecZ.z*cosT);

            handPointCloud.push_back(p);
        }

    }



    //3. Joints:
}














