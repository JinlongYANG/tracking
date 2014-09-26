#include "articulate_HandModel_XYZRGB.h"
#include <opencv2/calib3d/calib3d.hpp>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

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

float Distance(pcl::PointXYZRGB p1, pcl::PointXYZRGB p2){
    float dis = sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y)+(p1.z-p2.z)*(p1.z-p2.z));
    if (dis == 0)
        return 0.00000001;
    else
        return dis;
}

float Distance(pcl::PointXYZ p1, Mat p2){
    float dis = sqrt((p1.x-p2.at<float>(0,0))*(p1.x-p2.at<float>(0,0))+(p1.y-p2.at<float>(1,0))*(p1.y-p2.at<float>(1,0))+(p1.z-p2.at<float>(2,0))*(p1.z-p2.at<float>(2,0)));
    if (dis == 0)
        return 0.00000001;
    else
        return dis;
}

float Distance(pcl::PointXYZ p1, pcl::PointXYZRGB p2){
    float dis = sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y)+(p1.z-p2.z)*(p1.z-p2.z));
    if (dis == 0)
        return 0.00000001;
    else
        return dis;
}

float Distance(pcl::PointXYZ p1, pcl::PointXYZ p2){
    float dis = sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y)+(p1.z-p2.z)*(p1.z-p2.z));
    if (dis == 0)
        return 0.00000001;
    else
        return dis;
}

float Distance(Point3d p1, Point3d p2){
    float dis = sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y)+(p1.z-p2.z)*(p1.z-p2.z));
    if (dis == 0)
        return 0.00000001;
    else
        return dis;
}

float Distance(pcl::PointXYZ p1, Point3d p2){
    float dis = sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y)+(p1.z-p2.z)*(p1.z-p2.z));
    if (dis == 0)
        return 0.00000001;
    else
        return dis;
}

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
    bone_length[0][1] = 50.1779/1000;
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
    Model_joints[1].at<float>(0,0) = -0.014 - 0.002;
    Model_joints[1].at<float>(1,0) = -0.053;
    Model_joints[1].at<float>(2,0) = 0.002;
    //palm.index
    Model_joints[6].at<float>(0,0) = -0.014 - 0.002;
    Model_joints[6].at<float>(1,0) = -0.053;
    Model_joints[6].at<float>(2,0) = -0.008;

    Model_joints[7].at<float>(0,0) = -0.024 - 0.002;
    Model_joints[7].at<float>(1,0) = 0.019;
    Model_joints[7].at<float>(2,0) = 0;
    //palm.middle
    Model_joints[11].at<float>(0,0) = 0 - 0.002;
    Model_joints[11].at<float>(1,0) = -0.05;
    Model_joints[11].at<float>(2,0) = -0.008;

    Model_joints[12].at<float>(0,0) = -0.002 - 0.002;
    Model_joints[12].at<float>(1,0) = 0.023;
    Model_joints[12].at<float>(2,0) = -0.001;
    //palm.ring
    Model_joints[16].at<float>(0,0) = 0.014 - 0.002;
    Model_joints[16].at<float>(1,0) = -0.051;
    Model_joints[16].at<float>(2,0) = -0.008;

    Model_joints[17].at<float>(0,0) = 0.020 - 0.002;
    Model_joints[17].at<float>(1,0) = 0.018;
    Model_joints[17].at<float>(2,0) = 0.001;
    //palm.pinky
    Model_joints[21].at<float>(0,0) = 0.027 - 0.002;
    Model_joints[21].at<float>(1,0) = -0.053;
    Model_joints[21].at<float>(2,0) = -0.004;

    Model_joints[22].at<float>(0,0) = 0.042 - 0.002;
    Model_joints[22].at<float>(1,0) = 0.013;
    Model_joints[22].at<float>(2,0) = 0;

    for(int i = 0; i < 5; i++){
        virtual_joints[i] = Mat::zeros(3,1,CV_32FC1);
    }

    virtual_joints[0] = R_y(70)*Model_joints[1];

    virtual_joints[1].at<float>(0,0) = Model_joints[7].at<float>(0,0);
    virtual_joints[1].at<float>(1,0) = Model_joints[7].at<float>(1,0);
    virtual_joints[1].at<float>(2,0) = Model_joints[7].at<float>(2,0) + 0.1;

    virtual_joints[2].at<float>(0,0) = Model_joints[12].at<float>(0,0);
    virtual_joints[2].at<float>(1,0) = Model_joints[12].at<float>(1,0);
    virtual_joints[2].at<float>(2,0) = Model_joints[12].at<float>(2,0) + 0.1;

    virtual_joints[3].at<float>(0,0) = Model_joints[17].at<float>(0,0);
    virtual_joints[3].at<float>(1,0) = Model_joints[17].at<float>(1,0);
    virtual_joints[3].at<float>(2,0) = Model_joints[17].at<float>(2,0) + 0.1;

    virtual_joints[4].at<float>(0,0) = Model_joints[22].at<float>(0,0);
    virtual_joints[4].at<float>(1,0) = Model_joints[22].at<float>(1,0);
    virtual_joints[4].at<float>(2,0) = Model_joints[22].at<float>(2,0) + 0.1;

    //3.2.fingers:
    //3.2.1 index(extrinsic):
    Model_joints[8].at<float>(1,0) = bone_length[1][1];
    Model_joints[9].at<float>(1,0) = bone_length[1][2];
    Model_joints[10].at<float>(1,0) = bone_length[1][3];

    //    Model_joints[10] = Model_joints[10]+Model_joints[9]+Model_joints[8]+Model_joints[7];
    //    Model_joints[9] = Model_joints[9]+Model_joints[8]+Model_joints[7];
    //    Model_joints[8] = Model_joints[8]+Model_joints[7];

    //3.2.2 middel to pinky(extrinsic):
    for ( int i = 0; i < 3; ++i){
        Model_joints[i*5+13].at<float>(1,0) = bone_length[2+i][1];
        Model_joints[i*5+14].at<float>(1,0) = bone_length[2+i][2];
        Model_joints[i*5+15].at<float>(1,0) = bone_length[2+i][3];

        //        Model_joints[i*5+15] = Model_joints[i*5+15]+Model_joints[i*5+14]+Model_joints[i*5+13]+Model_joints[i*5+12];
        //        Model_joints[i*5+14] = Model_joints[i*5+14]+Model_joints[i*5+13]+Model_joints[i*5+12];
        //        Model_joints[i*5+13] = Model_joints[i*5+13]+Model_joints[i*5+12];

    }

    //3.2.3 thumb(extrinsic)
    Model_joints[2].at<float>(1,0) = bone_length[0][1];
    Model_joints[3].at<float>(1,0) = bone_length[0][2];
    Model_joints[4].at<float>(1,0) = bone_length[0][3];

    //    Model_joints[4] = Model_joints[4]+Model_joints[3]+Model_joints[2]+Model_joints[1];
    //    Model_joints[3] = Model_joints[3]+Model_joints[2]+Model_joints[1];
    //    Model_joints[2] = Model_joints[2]+Model_joints[1];
    //    Model_joints[4].copyTo(Model_joints[5]);

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


    //4 Point Cloud Model:
    //    float finger_angle_step = 30;
    //    float xy_relosution = 0.004;

    //4.1 Palm point cloud model:
    pcl::PointCloud<pcl::PointXYZRGB> palmPointCloud;
    pcl::PointXYZRGB palmpointXYZRGB;
    uint8_t r = 100;
    uint8_t g = 100;
    uint8_t b = 255;
    uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
    palmpointXYZRGB.rgb = *reinterpret_cast<float*>(&rgb);
    //length = y; width = x
    float length = -0.041;
    while(length < 0.038){
        float width = -0.030-(length/10.0);
        //left side:
        float hight_max = 0.01 + (0.04-length)/0.079*0.006;
        float hight = -hight_max;
        while(hight < hight_max){
            palmpointXYZRGB.x = width;
            palmpointXYZRGB.y = length-0.02;
            palmpointXYZRGB.z = hight;
            palmPointCloud.push_back(palmpointXYZRGB);
            hight += xy_relosution;
        }
        //upper and lower sides:
        while(width < 0.040+length/5.0){

            palmpointXYZRGB.x = width;
            palmpointXYZRGB.y = length-0.02;
            palmpointXYZRGB.z = hight_max;
            palmPointCloud.push_back(palmpointXYZRGB);

            palmpointXYZRGB.x = width;
            palmpointXYZRGB.y = length-0.02;
            palmpointXYZRGB.z = -hight_max;
            if(width <= 0)
                palmpointXYZRGB.z = -hight_max-0.005*(1+width/(0.030+length/10.0));
            else
                palmpointXYZRGB.z = -hight_max-0.005*(1-width/(0.040+length/5.0));
            palmPointCloud.push_back(palmpointXYZRGB);


            width += xy_relosution;
        }
        //right side:
        hight = -hight_max;
        while(hight < hight_max){
            palmpointXYZRGB.x = width;
            palmpointXYZRGB.y = length-0.02;
            palmpointXYZRGB.z = hight;
            palmPointCloud.push_back(palmpointXYZRGB);
            hight += xy_relosution;
        }

        length += xy_relosution;
    }
    //palm front
    float width = -0.030-(length/10.0);
    float hight_max = 0.01 + (0.04-length)/0.08*0.006;
    while(width < 0.040+length/5.0){
        float hight = 0;
        if(width <= 0)
            hight = -hight_max-0.005*(1+width/(0.030+length/10.0));
        else
            hight = -hight_max-0.005*(1-width/(0.040+length/5.0));
        while(hight < hight_max){
            palmpointXYZRGB.x = width;
            palmpointXYZRGB.y = length-0.02;
            palmpointXYZRGB.z = hight;
            palmPointCloud.push_back(palmpointXYZRGB);

            hight += xy_relosution;
        }
        width += xy_relosution;
    }

    handPointCloudVector.push_back(palmPointCloud);

    //4.2 finger point cloud model:
    for(int fingerIndex = 0; fingerIndex < 5; fingerIndex++){
        for(int boneIndex = 1; boneIndex < 4; boneIndex ++){
            pcl::PointCloud<pcl::PointXYZRGB> bonePointCloud;
            pcl::PointXYZRGB pointXYZRGB;
            r = fingerIndex*50+50;
            g = 255-(boneIndex * 80);
            b = 0;
            rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
            pointXYZRGB.rgb = *reinterpret_cast<float*>(&rgb);
            //joints:
            float theta = 0, phi = 0, joint_radius = 0.01-(boneIndex-1)*0.001;
            if (fingerIndex == 0 && boneIndex == 1)
                joint_radius = 0.015;
            if (fingerIndex == 0 && boneIndex == 2)
                joint_radius = 0.011;
            if (fingerIndex == 0 && boneIndex == 3)
                joint_radius = 0.009;
            if (fingerIndex == 4)
                joint_radius = joint_radius - 0.001;
            if( fingerIndex == 0 && boneIndex == 3){
                theta = 40;
                while(theta <= 90){
                    phi = 0;
                    while(phi < 360){
                        pointXYZRGB.x = joint_radius * cos(degree2arc(theta)) * cos(degree2arc(phi));
                        pointXYZRGB.y = joint_radius * cos(degree2arc(theta)) * sin(degree2arc(phi));
                        pointXYZRGB.z = joint_radius * sin(degree2arc(theta))*0.8;
                        bonePointCloud.push_back(pointXYZRGB);

                        phi += (30 + theta/2);
                    }
                    theta += 10;
                }
            }
            theta = 0;
            while(theta <= 90){
                phi = 0;
                while(phi < 360){
                    pointXYZRGB.x = joint_radius * cos(degree2arc(theta)) * cos(degree2arc(phi));
                    pointXYZRGB.y = joint_radius * cos(degree2arc(theta)) * sin(degree2arc(phi));
                    pointXYZRGB.z = -joint_radius * sin(degree2arc(theta))*0.8;
                    bonePointCloud.push_back(pointXYZRGB);

                    phi += (30 + theta/2);
                }
                theta += 10;
            }

            //fingers:
            float y = 0;
            float fingertip = 0;
            if (boneIndex == 3)
                fingertip = 0.006;
            while(y < bone_length[fingerIndex][boneIndex] - fingertip){
                pointXYZRGB.y = y;

                float angle = 0;
                float radius = 0.01-(boneIndex-1)*0.001-0.001/bone_length[fingerIndex][boneIndex]*y;
                if (fingerIndex == 0 && boneIndex == 1)
                    radius = 0.015-0.004/bone_length[fingerIndex][boneIndex]*y;
                if (fingerIndex == 0 && boneIndex == 2)
                    radius = 0.011-0.002/bone_length[fingerIndex][boneIndex]*y;
                if (fingerIndex == 0 && boneIndex == 3)
                    radius = 0.009-0.001/bone_length[fingerIndex][boneIndex]*y;
                if (fingerIndex == 4)
                    radius = radius - 0.001;
                while( angle < 360){
                    pointXYZRGB.x = radius*sin(degree2arc(angle));
                    if(fingerIndex == 0 && boneIndex == 3)
                        pointXYZRGB.z = radius*cos(degree2arc(angle))*(0.8-0.2*(1-(bone_length[0][3]- fingertip-y)/(bone_length[0][3]- fingertip)));
                    else
                        pointXYZRGB.z = radius*cos(degree2arc(angle))*0.9;
                    bonePointCloud.push_back(pointXYZRGB);
                    angle += finger_angle_step;

                }
                y += xy_relosution;
            }
            //fingertips
            if(boneIndex == 3){

                float radius = 0.007;
                if(fingerIndex == 0)
                    radius = 0.008;
                if(fingerIndex == 4)
                    radius = radius - 0.001;
                float phi = 0; theta = 0;
                while( theta <= 90){
                    phi = 0;
                    while( phi < 360 ){
                        pointXYZRGB.x = radius * cos(degree2arc(theta)) * cos(degree2arc(phi));
                        pointXYZRGB.y = radius * sin(degree2arc(theta)) + y;
                        if(fingerIndex == 0 && boneIndex == 3)
                            pointXYZRGB.z = radius * cos(degree2arc(theta)) * sin(degree2arc(phi)) *0.6;
                        else
                            pointXYZRGB.z = radius * cos(degree2arc(theta)) * sin(degree2arc(phi)) *0.9;
                        bonePointCloud.push_back(pointXYZRGB);

                        phi += (30 + theta/2);
                    }
                    theta += 10;
                }

            }

            handPointCloudVector.push_back(bonePointCloud);
        }
    }

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

    parameters_min[6] = -10;
    parameters_max[6] = 60;
    parameters_min[7] = 0;
    parameters_max[7] = 60;
    parameters_min[8] = -10;
    parameters_max[8] = 70;
    parameters_min[9] = -10;
    parameters_max[9] = 90;

    parameters_min[10] = -20;
    parameters_max[10] = 30;
    parameters_min[11] = -60;
    parameters_max[11] = 30;
    parameters_min[12] = -5;
    parameters_max[12] = 110;
    parameters_min[13] = 0;
    parameters_max[13] = 80;

    parameters_min[14] = -20;
    parameters_max[14] = 20;
    parameters_min[15] = -60;
    parameters_max[15] = 30;
    parameters_min[16] = -5;
    parameters_max[16] = 110;
    parameters_min[17] = 0;
    parameters_max[17] = 80;

    parameters_min[18] = -20;
    parameters_max[18] = 5;
    parameters_min[19] = -60;
    parameters_max[19] = 30;
    parameters_min[20] = -5;
    parameters_max[20] = 110;
    parameters_min[21] = 0;
    parameters_max[21] = 80;

    parameters_min[22] = -30;
    parameters_max[22] = 0;
    parameters_min[23] = -60;
    parameters_max[23] = 30;
    parameters_min[24] = -5;
    parameters_max[24] = 110;
    parameters_min[25] = 0;
    parameters_max[25] = 80;


    std::cout << "Model is ready!" << std::endl;

}

bool articulate_HandModel_XYZRGB::check_parameters(int &wrong_parameter_index){
    for(int i = 0; i < 26; i++){
        if(parameters[i] > parameters_max[i] || parameters[i] < parameters_min[i]){
            wrong_parameter_index = i;
            std::cout << "Wrong parameter index: " << wrong_parameter_index <<"; Value: "<< parameters[i] << std::endl;
            return false;
        }
    }
    return true;
}

void articulate_HandModel_XYZRGB::check_parameters(){
    for(int i = 0; i < 26; i++){
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
}


void articulate_HandModel_XYZRGB::set_parameters(float para[26]){
    for(int i = 0; i<26; i++)
        parameters[i] = para[i];
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
    //2.1 index(extrinsic):
    Mat R[3];
    R[0] = R_z(parameters[10])*R_x(parameters[11]);
    R[1] = R_x(parameters[12]);
    R[2] = R_x(parameters[13]);

    joints_for_calc[10] = R[0]*(R[1]*(R[2]*joints_for_calc[10]+joints_for_calc[9])+joints_for_calc[8])+joints_for_calc[7];
    joints_for_calc[9] = R[0]*(R[1]*joints_for_calc[9]+joints_for_calc[8])+joints_for_calc[7];
    joints_for_calc[8] = R[0]*joints_for_calc[8]+joints_for_calc[7];

    //2.2 middel to pinky(extrinsic):
    for ( int i = 0; i < 3; ++i){
        R[0] = R_z(parameters[i*4+14])*R_x(parameters[i*4+15]);
        R[1] = R_x(parameters[i*4+16]);
        R[2] = R_x(parameters[i*4+17]);

        joints_for_calc[i*5+15] = R[0]*(R[1]*(R[2]*joints_for_calc[i*5+15]+joints_for_calc[i*5+14])+joints_for_calc[i*5+13])+joints_for_calc[i*5+12];
        joints_for_calc[i*5+14] = R[0]*(R[1]*joints_for_calc[i*5+14]+joints_for_calc[i*5+13])+joints_for_calc[i*5+12];
        joints_for_calc[i*5+13] = R[0]*joints_for_calc[i*5+13]+joints_for_calc[i*5+12];

    }

    //2.3 thumb(extrinsic)
    R[0] = R_y(70);
    R[0] = R_z(10+parameters[6])*R_x(parameters[7])*R[0];
    R[1] = R_x(parameters[8]);
    R[2] = R_x(parameters[9]);

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

        for( int boneIndex = 0; boneIndex < 3; ++boneIndex){

            int vectorIndex = fingerIndex*3+1+boneIndex;
            temp_p.rgb = handPointCloudVector[vectorIndex].points[0].rgb;
            for(size_t i = 0; i < handPointCloudVector[vectorIndex].points.size(); ++i){
                Mat point4calcu = Mat::zeros(3,1,CV_32FC1);
                point4calcu.at<float>(0,0) = handPointCloudVector[vectorIndex].points[i].x;
                point4calcu.at<float>(1,0) = handPointCloudVector[vectorIndex].points[i].y;
                point4calcu.at<float>(2,0) = handPointCloudVector[vectorIndex].points[i].z;
                Mat R[3];
                R[0] = R_z(parameters[fingerIndex*4 + 6])*R_x(parameters[fingerIndex*4 + 7]);
                R[1] = R_x(parameters[fingerIndex*4 + 8]);
                R[2] = R_x(parameters[fingerIndex*4 + 9]);
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
    for( int boneIndex = 0; boneIndex < 3; ++boneIndex){
        int vectorIndex = boneIndex+1;
        temp_p.rgb = handPointCloudVector[vectorIndex].points[0].rgb;
        for(size_t i = 0; i < handPointCloudVector[vectorIndex].points.size(); ++i){
            Mat point4calcu = Mat::zeros(3,1,CV_32FC1);
            point4calcu.at<float>(0,0) = handPointCloudVector[vectorIndex].points[i].x;
            point4calcu.at<float>(1,0) = handPointCloudVector[vectorIndex].points[i].y;
            point4calcu.at<float>(2,0) = handPointCloudVector[vectorIndex].points[i].z;
            Mat R[3];
            R[0] = R_y(70);
            R[0] = R_z(10+parameters[6])*R_x(parameters[7])*R[0];
            R[1] = R_x(parameters[8]);
            R[2] = R_x(parameters[9]);
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
    for(size_t i = 0; i < handPointCloudVector[0].points.size(); ++i){
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
















