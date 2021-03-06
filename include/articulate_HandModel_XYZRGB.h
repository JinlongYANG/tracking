//****************************************************//
//************    Parameter illustration   ***********//
//26 angles corresponding to 26 degree of freedom
//0: hand center position x;
//1: hand center position y;
//2: hand center position z;

//3: palm rotation angle around x axis;
//4: palm rotation angle around y axis;
//5: palm rotation angle around z axis;

//6: horizontal angle between thumb metacarpal and palm;
//7: vertical angle between thumb metacarpal and palm;
//8: angle between thumb metacarpal and proximal;
//9: angle between thumb proximal and distal;

//10: horizontal angle between index finger proximal and palm;
//11: vertical angle between index finger proximal and palm;
//12: angle between index finger proximal and intermediate;
//13: angle between index finger intermediate and distal;

//14: horizontal angle between middle finger proximal and palm;
//15: vertical angle between middle finger proximal and palm;
//16: angle between middle finger proximal and intermediate;
//17: angle between middle finger intermediate and distal;

//18: horizontal angle between ring finger proximal and palm;
//19: vertical angle between ring finger proximal and palm;
//20: angle between ring finger proximal and intermediate;
//21: angle between ring finger intermediate and distal;

//22: horizontal angle between pinky proximal and palm;
//23: vertical angle between pinky proximal and palm;
//24: angle between pinky proximal and intermediate;
//25: angle between pinky intermediate and distal;

///////////////////////////////////////////////////////////
//************    Bone_length illustration  ***********//
//All with length unit of mm;

//[0][0]: thumb metacarpal length;
//[0][1]: thumb proximal length;
//[0][2]: thumb distal length;
//[0][3]: EMPTY;

//[1][0]: index finger metacarpal length;
//[1][1]: index finger proximal length;
//[1][2]: index finger intermediate length;
//[1][3]: index finger distal length;

//[2][0]: middle finger metacarpal length;
//[2][1]: middle finger proximal length;
//[2][2]: middle finger intermediate length;
//[2][3]: middle finger distal length;

//[3][0]: ring finger metacarpal length;
//[3][1]: ring finger proximal length;
//[3][2]: ring finger intermediate length;
//[3][3]: ring finger distal length;

//[4][0]: pinky finger metacarpal length;
//[4][1]: pinky finger proximal length;
//[4][2]: pinky finger intermediate length;
//[4][3]: pinky finger distal length;

////////////////////////////////////////////////////////////////
//***********     Joints_position illustration   **************//
//0: palm center;

//1: begin of thumb metacarpal;
//2: end of thumb metacarpal/begin of index finger proximal;
//3: end of thumb proximal/begin of index finger distal;
//4: end of thumb distal;
//5: end of thumb distal;

//6: begin of index finger metacarpal;
//7: end of index finger metacarpal/begin of index finger proximal;
//8: end of index finger proximal/begin of index finger intermediate;
//9: end of index finger intermediate/begin of index finger distal;
//10: end of index finger distal;

//11: begin of middle finger metacarpal;
//12: end of middle finger metacarpal/begin of index finger proximal;
//13: end of middle finger proximal/begin of index finger intermediate;
//14: end of middle finger intermediate/begin of index finger distal;
//15: end of middle finger distal;

//16: begin of ring finger metacarpal;
//17: end of ring finger metacarpal/begin of index finger proximal;
//18: end of ring finger proximal/begin of index finger intermediate;
//19: end of ring finger intermediate/begin of index finger distal;
//20: end of ring finger distal;

//21: begin of pinky finger metacarpal;
//22: end of pinky finger metacarpal/begin of index finger proximal;
//23: end of pinky finger proximal/begin of index finger intermediate;
//24: end of pinky finger intermediate/begin of index finger distal;
//25: end of pinky finger distal;

////////////////////////////////////////////////////////////////////


#ifndef articulate_HandModel_XYZRGB_H
#define articulate_HandModel_XYZRGB_H

#include <opencv2/core/core.hpp>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <omp.h>

using namespace cv;



class articulate_HandModel_XYZRGB
{
public:
    articulate_HandModel_XYZRGB(const float finger_angle_step = 15, const float xy_relosution = 0.003);

    float parameters[27];
    float parameters_max[27];
    float parameters_min[27];
    float bone_length[5][4];
    pcl::PointXYZRGB joints_position[26];
    Mat Model_joints[26];
    Mat auxiliary_palm_position[10];
    Mat auxiliary_palm_position_now[10];
    Mat palm_model;
    vector< pcl::PointCloud<pcl::PointXYZRGB> > handPointCloudVector;


    bool check_parameters(int & wrong_parameter_index);
    void check_parameters();
    void set_parameters();
    void set_parameters(float para[27]);
    void set_parameters(vector<float> para);
    void get_joints_positions();
    void get_handPointCloud(pcl::PointCloud<pcl::PointXYZRGB> & handPointCloud);
    void samplePointCloud(pcl::PointCloud<pcl::PointXYZRGB> & handPointCloud);


private:


};

#endif // articulate_HandModel_XYZRGB_H

