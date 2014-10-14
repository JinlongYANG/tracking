#define BOOST_SIGNALS_NO_DEPRECATION_WARNING
#include "tracking_node.hpp"
#include "handkp_leap_msg.h"
//#include "tracking/Hand_XYZRGB.h"
#include "articulate_HandModel_XYZRGB.h"
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <math.h>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <omp.h>
#include "para_deque.h"

#define PI 3.14159265
#define PARTICLE_NUM 8

using namespace image_transport;
using namespace sensor_msgs;
using namespace geometry_msgs;
using namespace cv;
using namespace image_geometry;
using namespace pcl;
using namespace Eigen;
using namespace std;
//using namespace tf2;

float half_palm_length = 0.07;
float finger_length = 0.07;
int resolution = 3;
//30cm *30cm for hand
int imageSize = 300/resolution;
bool flag_leap = true;
//pcl::PointXYZRGB my_hand_kp[31];
//pcl::PointXYZRGB last_joints_position[26];
float para_lastFrame[27], det_para[27];
int seq = 0;
int back_groud_value = 100;
para_deque parameter_que[PARTICLE_NUM];
para_deque best_parameter_particle;

articulate_HandModel_XYZRGB MyHand;

tracking_Node::tracking_Node(ros::NodeHandle& nh):
    imageTransport_(nh),
    timeSynchronizer_(10)
  //reconfigureServer_(ros::NodeHandle(nh,"tracking")),
  //transformListener_(buffer_, true)
  //reconfigureCallback_(boost::bind(&tracking_Node::updateConfig, this, _1, _2))
{

    hand_kp_Subscriber_.subscribe(nh, "/Hand_kp_cl", 10);
    hand_Subscriber_.subscribe(nh, "/Hand_pcl", 10);
    registered_Depth_Image_Subscriber.subscribe(nh, "/Depth_Image", 10);

    timeSynchronizer_.connectInput(hand_kp_Subscriber_, hand_Subscriber_, registered_Depth_Image_Subscriber);

    handPublisher_ = nh.advertise<sensor_msgs::PointCloud2>("Hand_cloud",0);
    articulatePublisher_ = nh.advertise<sensor_msgs::PointCloud2>("Articulate",0);
    modelPublisher_ = nh.advertise<sensor_msgs::PointCloud2>("Model_point_cloud",0);

    bone_pub_ = nh.advertise<visualization_msgs::Marker>("Bones", 0);
    bone_leap_pub_ = nh.advertise<visualization_msgs::Marker>("Leap_Articulate", 0);
    //leap_articulate_pub_ = nh.advertise<visualization_msgs::Marker>("Leap_Articulate",0);

    timeSynchronizer_.registerCallback(boost::bind(&tracking_Node::syncedCallback, this, _1, _2, _3));

    srand( (unsigned)clock( ) );
    //reconfigureServer_.setCallback(reconfigureCallback_);


}

//void  SlamNode::updateConfig(pixel_slam::slamConfig &config, uint32_t level){
//    slam_.reset(new Slam(config.min_depth,config.max_depth,config.line_requirement));
//    line_requirement_=config.line_requirement;
//    if(stereoCameraModel_.initialized()){
//        min_disparity_=stereoCameraModel_.getDisparity(config.max_depth);
//        max_disparity_=stereoCameraModel_.getDisparity(config.min_depth);
//    }
//}

void tracking_Node::syncedCallback(const PointCloud2ConstPtr& hand_kp_pter, const PointCloud2ConstPtr& hand_pter, const ImageConstPtr& cvpointer_depthImage){

    ros::Time time0 = ros::Time::now();
    cv_bridge::CvImagePtr cvpointer_depthFrame;

    int tracking_mode = 2;

    pcl::PointCloud<pcl::PointXYZRGB> msg_pcl, handcloud, hand_kp;

    ROS_INFO("Callback begins");

    try
    {
        //********************************************************//
        //1.1 get ready of Hand point cloud and joints position
        // hand point cloud in pcl::PointCloud<pcl::PointXYZRGB> handcloud; and joints position in: hand_kp;
        ros::Time time1 = ros::Time::now();

        fromROSMsg(*hand_kp_pter, hand_kp);
        fromROSMsg(*hand_pter, msg_pcl);

        //std::cout<<"Hand Keypoint size: "<<hand1_kp.size()<<endl;
        //std::cout<<"Hand Cloud size: "<<msg_pcl.size()<<endl;

        cvpointer_depthFrame = cv_bridge::toCvCopy(cvpointer_depthImage);
        Mat Origional_depthImage;
        Origional_depthImage = cvpointer_depthFrame->image;

        ros::Time time2 = ros::Time::now();

        for (size_t i = 0; i < msg_pcl.points.size (); ++i){
            if( (abs(msg_pcl.points[i].x) < 0.2
                 && abs(msg_pcl.points[i].y) < 0.2
                 && abs(msg_pcl.points[i].z) < 0.2)
                    &&((abs(msg_pcl.points[i].x)*abs(msg_pcl.points[i].x)+ abs(msg_pcl.points[i].y)*abs(msg_pcl.points[i].y)+abs(msg_pcl.points[i].z)*abs(msg_pcl.points[i].z)< half_palm_length*half_palm_length) ||
                       (abs(msg_pcl.points[i].x - hand_kp.points[1].x) < finger_length
                        && abs(msg_pcl.points[i].y - hand_kp.points[1].y) < finger_length
                        && abs(msg_pcl.points[i].z - hand_kp.points[1].z) < finger_length)||
                       (abs(msg_pcl.points[i].x - hand_kp.points[7].x) < finger_length
                        && abs(msg_pcl.points[i].y - hand_kp.points[7].y) < finger_length
                        && abs(msg_pcl.points[i].z - hand_kp.points[7].z) < finger_length)||
                       (abs(msg_pcl.points[i].x - hand_kp.points[13].x) < finger_length
                        && abs(msg_pcl.points[i].y - hand_kp.points[13].y) < finger_length
                        && abs(msg_pcl.points[i].z - hand_kp.points[13].z) < finger_length)||
                       (abs(msg_pcl.points[i].x - hand_kp.points[19].x) < finger_length
                        && abs(msg_pcl.points[i].y - hand_kp.points[19].y) < finger_length
                        && abs(msg_pcl.points[i].z - hand_kp.points[19].z) < finger_length)||
                       (abs(msg_pcl.points[i].x - hand_kp.points[25].x) < finger_length
                        && abs(msg_pcl.points[i].y - hand_kp.points[25].y) < finger_length
                        && abs(msg_pcl.points[i].z - hand_kp.points[25].z) < finger_length))
                    ){
                //                uint32_t rgb = *reinterpret_cast<int*>(&msg_pcl.points[i].rgb);
                //                uint8_t r = (rgb >> 16) & 0x0000ff;
                //                uint8_t g = (rgb >> 8) & 0x0000ff;
                //                uint8_t b = (rgb) & 0x0000ff;
                //                if(r > b && r > 70){

                handcloud.push_back(msg_pcl.points[i]);
            }
        }

        ros::Time time3 = ros::Time::now();
        //******** 1.1 done  **************//

        //********************************************************//
        //1.2 Do segmentation

        //******** 1.2 done  **************//
        ros::Time time4 = ros::Time::now();


        //******** 1.3 Ray tracing for Oberservation **********//
        // Determin whether point in visible or not, and generate visibility map stored in visibilityMap_Oberservation and visiblityMap_Hypo;
        Mat visibilityMap_Oberservation(imageSize,imageSize,CV_8UC1,Scalar(back_groud_value));
        Ray_tracing_OrthognalProjection(handcloud, imageSize, resolution, visibilityMap_Oberservation);
        //******** 1.3 done *****************//
        ros::Time time5 = ros::Time::now();
        std::cout << time1-time0 << " seconds: " << "from callback begin to try." << std::endl;
        std::cout << time2-time1 << " seconds: " <<"message from transfer." <<std::endl;
        std::cout << time3-time2 << " seconds: " << "Picking hand points." << std::endl;
        std::cout << time5-time4 << " seconds: Ray tracking for Observation." << std::endl;

        pcl::PointCloud<pcl::PointXYZRGB> modelPointCloud;

        //******** 2.0 Point cloud hand model visulization *********//
        if(tracking_mode == -1){
            float para[27];
            float para_suboptimal[27];
            float temp_parameters[27]= {0,0,0,
                                        0,0,0,
                                        70,-40,60,20,
                                        30,0,10,10,
                                        0,10,20,10,
                                        -10,60,60,20,
                                        -20,70,70,10,
                                        70};

            for(int i = 0; i<27;i++){
                para_lastFrame[i] = temp_parameters[i];
                det_para[i] = 0;
                para[i] = temp_parameters[i];
                para_suboptimal[i] = temp_parameters[i];
            }
            MyHand.set_parameters(para);
            MyHand.get_joints_positions();
            ros::Time time_begin = ros::Time::now();
            //MyHand.get_handPointCloud(modelPointCloud);
            ros::Time time_end = ros::Time::now();
            //std::cout << time_end-time_begin << " seconds: Transform point cloud." <<std::endl;
            time_begin = ros::Time::now();
            MyHand.samplePointCloud(modelPointCloud);
            time_end = ros::Time::now();
            std::cout << time_end-time_begin << " seconds: Resample point cloud." <<std::endl;

            //******** 2.1 Projection(visibility map) ******//
            time_begin = ros::Time::now();
            Mat visiblityMap_Hypo(imageSize,imageSize,CV_8UC1,Scalar(back_groud_value));
            Ray_tracing_OrthognalProjection(modelPointCloud, imageSize, resolution, visiblityMap_Hypo);
            time_end = ros::Time::now();
            std::cout << time_end-time_begin << " seconds: Projection of  point cloud." <<std::endl;
            //******** 2.1 done ******//

            //******** 2.2 Score (similarity assessment) ******//
            time_begin = ros::Time::now();
            float overlap, overall_diff, overlap_diff, overlap_obs, overlap_hyp;
            Score(visibilityMap_Oberservation, visiblityMap_Hypo, back_groud_value, overlap, overlap_obs, overlap_hyp, overall_diff, overlap_diff);
            time_end = ros::Time::now();
            std::cout << time_end-time_begin << " seconds: Scoring." <<std::endl;
            //******** 2.2 done *************//
        }

        //******** 2.0 particle filters for tracking --------ONE particle****//
        else if (tracking_mode == 0){
            float optimal_para[27];
            float Opt_Score_overlapratio1 = 100000, Opt_Score_aver_overlapdiff = 100000, Opt_Score_overlapratio2 = 10000;
            for(int annealing_iterator = 0; annealing_iterator < 3; annealing_iterator++){
                float annealing_factor = pow(0.6, annealing_iterator);
                for(int parameterDimension = -1; parameterDimension < 27; ++parameterDimension){
                    float para[27];
                    float para_suboptimal[27];

                    //very first (initialization of the whole programme)
                    if((!seq)&&(!annealing_iterator)&&(parameterDimension == -1)){
                        float temp_parameters[27]= {0,0,0,
                                                    -30,0,-10,
                                                    10,-30,0,10,
                                                    10,0,0,0,
                                                    0,0,0,0,
                                                    -10,0,0,0,
                                                    -20,0,0,0,
                                                    70};

                        for(int i = 0; i<27;i++){
                            para_lastFrame[i] = temp_parameters[i];
                            det_para[i] = 0;
                            para[i] = temp_parameters[i];
                            para_suboptimal[i] = temp_parameters[i];
                        }
                    }
                    //use last frame result for current frame initialization
                    else if ((!annealing_iterator)&&(parameterDimension == -1)){
                        for(int i = 0; i<27;i++){
                            para[i] = parameter_que[0].para_sequence_smoothed[0][i] + det_para[i];
                            para_suboptimal[i] = para[i];
                        }
                    }

                    //use last kinematic_chain result for current kinematic chain initialization
                    else{
                        for(int i = 0; i<27; ++i){
                            para[i] = optimal_para[i];
                            para_suboptimal[i] = para[i];
                        }
                    }
                    int max_iterator = 4;

                    float translation_step = 0.01;
                    int translation_mode = int(2*translation_step*1000);

                    float angle_step = 10;
                    int angle_mode = int(2*angle_step*100);

                    //#pragma omp for
                    for (int iterator = 0; iterator < max_iterator; iterator ++){
                        //******** 2.1 generate Hypothesis point cloud *******//
                        if(parameterDimension == -1){
                            max_iterator = 1;
                        }
                        else if(parameterDimension < 3){
                            para[parameterDimension] += (rand()%translation_mode/1000.0/max_iterator+2*translation_step/max_iterator*iterator-translation_step)*annealing_factor;
                        }
                        else if (parameterDimension < 5){
                            para[parameterDimension] += (rand()%angle_mode/100.0/max_iterator+2*angle_step/max_iterator*iterator-angle_step)*annealing_factor;
                            if(parameterDimension == 4){

                            }
                        }
                        else if (parameterDimension == 5){
                            //para[5] += ((para[10]-10)+(para[14]-0)+(para[18]+6.7))/3.0;
                            para[5] += (rand()%angle_mode/100.0/max_iterator+2*angle_step/max_iterator*iterator-angle_step)*annealing_factor;
                        }
                        else if (parameterDimension == 6){
                            para[6] += (rand()%angle_mode/100.0/max_iterator+2*angle_step/max_iterator*iterator-angle_step)*annealing_factor*2;
                        }
                        else if (parameterDimension == 7){
                            para[7] += (rand()%angle_mode/100.0/max_iterator+2*angle_step/max_iterator*iterator-angle_step)*annealing_factor*2;
                        }
                        else if (parameterDimension == 8){
                            para[8] += (rand()%angle_mode/100.0/max_iterator+2*angle_step/max_iterator*iterator-angle_step)*annealing_factor;
                        }
                        else if ( parameterDimension == 26){
                            para[26] += (rand()%angle_mode/100.0/max_iterator+2*angle_step/max_iterator*iterator-angle_step)*annealing_factor/2.0;
                        }
                        else if ( (parameterDimension -6)%4 == 0){
                            para[parameterDimension] += (rand()%angle_mode/100.0/max_iterator+2*angle_step/max_iterator*iterator-angle_step)*annealing_factor/2.0;
                        }
                        else{
                            para[parameterDimension] += (rand()%angle_mode/100.0/max_iterator+2*angle_step/max_iterator*iterator-angle_step)*annealing_factor*3.0;
                        }

                        //check parameters:
                        {
                            //min and max
                            for(int i = 0; i < 27; i++){
                                if(para[i] > MyHand.parameters_max[i]){
                                    para[i] = MyHand.parameters_max[i];
                                }
                                else if (para[i] < MyHand.parameters_min[i]){
                                    para[i] = MyHand.parameters_min[i];
                                }
                            }
                            //collision between 2~5 finger
                            for(int i = 10; i < 19; i += 4){
                                if(fabs(para[i+1]-para[i+5]) < 20){
                                    if(para[i] - para[i+4] < -5){
                                        float dif = para[i+4] - para[i];
                                        para[i+4] -= dif;
                                        para[i] += dif;
                                    }
                                }
                            }
                        }
                        //generate hypo
                        MyHand.set_parameters(para);
                        MyHand.get_joints_positions();
                        //ros::Time time_begin = ros::Time::now();
                        //MyHand.get_handPointCloud(modelPointCloud);
                        MyHand.samplePointCloud(modelPointCloud);
                        //ros::Time time_end = ros::Time::now();
                        //std::cout<<"Duration: "<< time_end-time_begin << std::endl;
                        //******** 2.1 done ****************//

                        //******** 2.2 Ray tracing for Hypothesis ********//
                        Mat visiblityMap_Hypo(imageSize,imageSize,CV_8UC1,Scalar(back_groud_value));
                        pcl::PointCloud<pcl::PointXYZRGB> visibleModelPointCloud;
                        //Ray_tracing_OrthognalProjection(modelPointCloud, imageSize, resolution, visiblityMap_Hypo, visibleModelPointCloud);
                        Ray_tracing_OrthognalProjection(modelPointCloud, imageSize, resolution, visiblityMap_Hypo);
                        //******** 2.2 done *******************//

                        //        imshow("visibilityMap_Oberservation", visibilityMap_Oberservation);
                        //        imshow("visiblityMap_Hypo", visiblityMap_Hypo);
                        //        waitKey();

                        //ROS_INFO("Prepare Model Cloud");
                        //                        sensor_msgs::PointCloud2 model_cloud_msg;
                        //                        toROSMsg(visibleModelPointCloud,model_cloud_msg);
                        //                        model_cloud_msg.header.frame_id=hand_kp_pter->header.frame_id;
                        //                        model_cloud_msg.header.stamp = hand_kp_pter->header.stamp;
                        //                        modelPublisher_.publish(model_cloud_msg);

                        //******** 2.3 Score (similarity assessment) ******//
                        float overlap, overall_diff, overlap_diff, overlap_obs, overlap_hyp;
                        Score(visibilityMap_Oberservation, visiblityMap_Hypo, back_groud_value, overlap, overlap_obs, overlap_hyp, overall_diff, overlap_diff);
                        //******** 2.3 done *************//

                        //std::cout << "Overall_diff: " << overall_diff << std::endl;

                        if((overlap_obs/overlap <= Opt_Score_overlapratio1 || overlap_hyp/overlap <= Opt_Score_overlapratio2) && (overlap_diff/overlap <= Opt_Score_aver_overlapdiff || overlap_diff/overlap <= 1.2)){
                            Opt_Score_overlapratio1 = min(overlap_obs/overlap,Opt_Score_overlapratio1);
                            Opt_Score_overlapratio2 = min(overlap_hyp/overlap, Opt_Score_overlapratio2);
                            if(overlap_diff/overlap <= Opt_Score_aver_overlapdiff )
                                Opt_Score_aver_overlapdiff = 1.0*overlap_diff/overlap;
                            for(int i = 0; i< 27; i++){
                                optimal_para[i] = para[i];
                            }
                        }


                        for(int i = 0; i< 27; i++){
                            //std::cout << "para" << i <<": " << para[i] << std::endl;
                            para[i] = para_suboptimal[i];
                        }
                    }
                }
            }
            vector<float> for_the_que;
            for(int i = 0; i< 27; i++){
                std::cout << "para" << i <<": " << optimal_para[i] << std::endl;
                para_lastFrame[i] = optimal_para[i];
                for_the_que.push_back(optimal_para[i]);
            }
            parameter_que[0].add_new(for_the_que);
            parameter_que[0].smooth_mean(3);
            for(int i = 0; i< 27; i++){
                det_para[i] = parameter_que[0].para_delta[i];
            }

            std::cout << "Overlap ratio1: " <<Opt_Score_overlapratio1 << std::endl;
            std::cout << "Overlap ratio2: " <<Opt_Score_overlapratio2 << std::endl;
            std::cout << "Average overlap distance: " << Opt_Score_aver_overlapdiff << std::endl;

            for(int i = 0; i< 27; i++){
                optimal_para[i] = parameter_que[0].para_sequence_smoothed[0][i];
            }
            MyHand.set_parameters(optimal_para);
            MyHand.get_joints_positions();
            //MyHand.get_handPointCloud(modelPointCloud);
            MyHand.samplePointCloud(modelPointCloud);
        }

        //******** 2.0 particle filters for tracking --------MULTI particles*****//
        else if (tracking_mode == 1){
            Point3d errors[PARTICLE_NUM];

            //           omp_set_num_threads(4);
            //            #pragma omp parallel for
            for(int particle_index = 0; particle_index < PARTICLE_NUM; particle_index ++){
                //articulate_HandModel_XYZRGB openMP_hand;
                float optimal_para[27];
                float Opt_Score_overlapratio1 = 100000, Opt_Score_aver_overlapdiff = 100000, Opt_Score_overlapratio2 = 10000;
                for(int annealing_iterator = 0; annealing_iterator < 3; annealing_iterator++){
                    float annealing_factor = pow(0.6, annealing_iterator);
                    for(int parameterDimension = -1; parameterDimension < 27; ++parameterDimension){
                        float para[27];
                        float para_suboptimal[27];

                        //very first (initialization of the whole programme)
                        if((!seq)&&(!annealing_iterator)&&(parameterDimension == -1)){
                            float temp_parameters[27]= {0,0,0,
                                                        -30,0,-10,
                                                        10,-30,0,10,
                                                        10,0,0,0,
                                                        0,0,0,0,
                                                        -10,0,0,0,
                                                        -20,0,0,0,
                                                        70};

                            for(int i = 0; i<27;i++){
                                para_lastFrame[i] = temp_parameters[i];
                                det_para[i] = 0;
                                para[i] = temp_parameters[i];
                                para_suboptimal[i] = temp_parameters[i];
                            }
                        }
                        //use last frame result for current frame initialization
                        else if ((!annealing_iterator)&&(parameterDimension == -1)){
                            for(int i = 0; i<27;i++){
                                para[i] = parameter_que[particle_index].para_sequence_smoothed[0][i] + det_para[i];
                                para_suboptimal[i] = para[i];
                            }
                        }

                        //use last kinematic_chain result for current kinematic chain initialization
                        else{
                            for(int i = 0; i<27; ++i){
                                para[i] = optimal_para[i];
                                para_suboptimal[i] = para[i];
                            }
                        }
                        int max_iterator = 4;

                        float translation_step = 0.01;
                        int translation_mode = int(2*translation_step*1000);

                        float angle_step = 10;
                        int angle_mode = int(2*angle_step*100);

                        for (int iterator = 0; iterator < max_iterator; iterator ++){
                            //******** 2.1 generate Hypothesis point cloud *******//
                            if (parameterDimension == -1){
                                max_iterator = 1;
                            }
                            else if(parameterDimension < 3){
                                para[parameterDimension] += (rand()%translation_mode/1000.0/max_iterator+2*translation_step/max_iterator*iterator-translation_step)*annealing_factor;
                            }
                            else if (parameterDimension < 5){
                                para[parameterDimension] += (rand()%angle_mode/100.0/max_iterator+2*angle_step/max_iterator*iterator-angle_step)*annealing_factor;
                                if(parameterDimension == 4){

                                }
                            }
                            else if (parameterDimension == 5){
                                para[5] += ((para[10]-10)+(para[14]-0)+(para[18]+6.7))/3.0;
                                para[5] += (rand()%angle_mode/100.0/max_iterator+2*angle_step/max_iterator*iterator-angle_step)*annealing_factor;
                            }
                            else if (parameterDimension == 6){
                                para[6] += (rand()%angle_mode/100.0/max_iterator+2*angle_step/max_iterator*iterator-angle_step)*annealing_factor;
                            }
                            else if (parameterDimension == 7){
                                para[7] += (rand()%angle_mode/100.0/max_iterator+2*angle_step/max_iterator*iterator-angle_step)*annealing_factor*2;
                            }
                            else if (parameterDimension == 8){
                                para[8] += (rand()%angle_mode/100.0/max_iterator+2*angle_step/max_iterator*iterator-angle_step)*annealing_factor*3;
                            }
                            else if ( parameterDimension == 26){
                                para[26] += (rand()%angle_mode/100.0/max_iterator+2*angle_step/max_iterator*iterator-angle_step)*annealing_factor/2.0;
                            }
                            else if ( (parameterDimension -6)%4 == 0){
                                para[parameterDimension] += (rand()%angle_mode/100.0/max_iterator+2*angle_step/max_iterator*iterator-angle_step)*annealing_factor/2.0;
                            }
                            else{
                                para[parameterDimension] += (rand()%angle_mode/100.0/max_iterator+2*angle_step/max_iterator*iterator-angle_step)*annealing_factor*3.0;
                            }

                            //check parameters:
                            {
                                //min and max
                                for(int i = 0; i < 27; i++){
                                    if(para[i] > MyHand.parameters_max[i]){
                                        para[i] = MyHand.parameters_max[i];
                                    }
                                    else if (para[i] < MyHand.parameters_min[i]){
                                        para[i] = MyHand.parameters_min[i];
                                    }
                                }
                                //collision between 2~5 finger
                                for(int i = 10; i < 19; i += 4){
                                    if(fabs(para[i+1]-para[i+5]) < 20){
                                        if(para[i] - para[i+4] < -5){
                                            float dif = para[i+4] - para[i];
                                            para[i+4] -= dif;
                                            para[i] += dif;
                                        }
                                    }
                                }
                            }

                            //generate hypo
                            MyHand.set_parameters(para);
                            MyHand.get_joints_positions();
                            //ros::Time time_begin = ros::Time::now();
                            //MyHand.get_handPointCloud(modelPointCloud);
                            MyHand.samplePointCloud(modelPointCloud);
                            //ros::Time time_end = ros::Time::now();
                            //std::cout<<"Duration: "<< time_end-time_begin << std::endl;
                            //******** 2.1 done ****************//

                            //******** 2.2 Ray tracing for Hypothesis ********//
                            Mat visiblityMap_Hypo(imageSize,imageSize,CV_8UC1,Scalar(back_groud_value));
                            pcl::PointCloud<pcl::PointXYZRGB> visibleModelPointCloud;
                            //Ray_tracing_OrthognalProjection(modelPointCloud, imageSize, resolution, visiblityMap_Hypo, visibleModelPointCloud);
                            Ray_tracing_OrthognalProjection(modelPointCloud, imageSize, resolution, visiblityMap_Hypo);
                            //******** 2.2 done *******************//

                            //******** 2.3 Score (similarity assessment) ******//
                            float overlap, overall_diff, overlap_diff, overlap_obs, overlap_hyp;
                            Score(visibilityMap_Oberservation, visiblityMap_Hypo, back_groud_value, overlap, overlap_obs, overlap_hyp, overall_diff, overlap_diff);
                            //******** 2.3 done *************//

                            //std::cout << "Overall_diff: " << overall_diff << std::endl;

                            if((overlap_obs/overlap <= Opt_Score_overlapratio1 || overlap_hyp/overlap <= Opt_Score_overlapratio2) && (overlap_diff/overlap <= Opt_Score_aver_overlapdiff || overlap_diff/overlap <= 1.2)){
                                Opt_Score_overlapratio1 = min(overlap_obs/overlap,Opt_Score_overlapratio1);
                                Opt_Score_overlapratio2 = min(overlap_hyp/overlap, Opt_Score_overlapratio2);
                                if(overlap_diff/overlap <= Opt_Score_aver_overlapdiff )
                                    Opt_Score_aver_overlapdiff = 1.0*overlap_diff/overlap;
                                for(int i = 0; i< 27; i++){
                                    optimal_para[i] = para[i];
                                }
                            }

                            for(int i = 0; i< 27; i++){
                                //std::cout << "para" << i <<": " << para[i] << std::endl;
                                para[i] = para_suboptimal[i];
                            }
                        }
                    }
                }

                vector<float> for_the_que;
                for(int i = 0; i< 27; i++){
                    std::cout << "para" << i <<": " << optimal_para[i] << std::endl;
                    para_lastFrame[i] = optimal_para[i];
                    for_the_que.push_back(optimal_para[i]);
                }
                parameter_que[particle_index].add_new(for_the_que);
                parameter_que[particle_index].smooth_mean(3);
                errors[particle_index].x = Opt_Score_overlapratio1;
                errors[particle_index].y = Opt_Score_overlapratio2;
                errors[particle_index].z = Opt_Score_aver_overlapdiff;
            }

            //Find the best particle out of all
            int best_particle_index = 0, worst_particle_index = 0;
            float weight_a = 2.0, weight_b = 2.0, weight_c = 1.0;
            float best_particle_error = weight_a * errors[0].x + weight_b * errors[0].y + weight_c * errors[0].z;
            float worst_particle_error = weight_a * errors[0].x + weight_b * errors[0].y + weight_c * errors[0].z;
            for(int particle_index = 1; particle_index < PARTICLE_NUM; particle_index++){
                float score = weight_a * errors[particle_index].x+weight_b * errors[particle_index].y+weight_c * errors[particle_index].z;
                if(score<best_particle_error){
                    best_particle_error = score;
                    best_particle_index = particle_index;
                }
                if(score > worst_particle_error){
                    worst_particle_error = score;
                    worst_particle_index = particle_index;
                }
            }
            std::cout<<"Best particle index: " << best_particle_index << std::endl;
            std::cout<<"Best particle error: " << errors[best_particle_index].x << ", " <<
                       errors[best_particle_index].y << ", " << errors[best_particle_index].z <<
                       " " << best_particle_error<<std::endl;

            std::cout<<"Worst particle index: " << worst_particle_index << std::endl;
            std::cout<<"Worst particle error: " << errors[worst_particle_index].x << ", " <<
                       errors[worst_particle_index].y << ", " << errors[worst_particle_index].z <<
                       " " << worst_particle_error<<std::endl;

            //reset the worst:
            for(int i = 0; i< 27; i++){
                parameter_que[worst_particle_index].para_sequence_smoothed[0][i] = parameter_que[best_particle_index].para_sequence_smoothed[0][i];
            }

            best_parameter_particle.add_new(parameter_que[best_particle_index].para_sequence_smoothed[0]);
            best_parameter_particle.smooth_mean(3);
            for(int i = 0; i< 27; i++){
                det_para[i] = best_parameter_particle.para_delta[i];
            }

            //Calculate best hand/finger pose:
            MyHand.set_parameters(parameter_que[best_particle_index].para_sequence_smoothed[0]);
            MyHand.get_joints_positions();
            MyHand.samplePointCloud(modelPointCloud);

        }

        //******** 2.0 particle filters for tracking --------MULTI particles with OpenMP*****//
        else if (tracking_mode == 2){
            Point3d errors[PARTICLE_NUM];

            ros::Time thread_begin = ros::Time::now();

            omp_set_num_threads(PARTICLE_NUM);
#pragma omp parallel
            {
                int  id = omp_get_thread_num();
                ros::Time thread_assigned = ros::Time::now();
                articulate_HandModel_XYZRGB openMP_hand;
                ros::Time thread_handModel = ros::Time::now();
                pcl::PointCloud<pcl::PointXYZRGB> openMP_modelPointCloud;
                float optimal_para[27];
                float Opt_Score_overlapratio1 = 100000, Opt_Score_aver_overlapdiff = 100000, Opt_Score_overlapratio2 = 10000;
                for(int annealing_iterator = 0; annealing_iterator < 3; annealing_iterator++){
                    float annealing_factor = pow(0.6, annealing_iterator);
                    int parameterDimension = 0;
                    if (!annealing_factor)
                        parameterDimension = -1;
                    for(; parameterDimension < 27; ++parameterDimension){
                        float para[27];
                        float para_suboptimal[27];

                        //very first (initialization of the whole programme)
                        if((!seq)&&(!annealing_iterator)&&(parameterDimension == -1)){
                            float temp_parameters[27]= {0,0,0,
                                                        -30,0,-10,
                                                        10,-30,0,10,
                                                        10,0,0,0,
                                                        0,0,0,0,
                                                        -10,0,0,0,
                                                        -20,0,0,0,
                                                        70};

                            for(int i = 0; i<27;i++){
                                para[i] = temp_parameters[i];
                                para_suboptimal[i] = temp_parameters[i];
                            }
                        }
                        //use last frame result for current frame initialization
                        else if ((!annealing_iterator)&&(parameterDimension == -1)){
                            for(int i = 0; i<27;i++){
                                para[i] = parameter_que[id].para_sequence_smoothed[0][i] + det_para[i];
                                para_suboptimal[i] = para[i];
                            }
                        }

                        //use last kinematic_chain result for current kinematic chain initialization
                        else{
                            for(int i = 0; i<27; ++i){
                                para[i] = optimal_para[i];
                                para_suboptimal[i] = para[i];
                            }
                        }
                        int max_iterator = 4;

                        float translation_step = 0.01;
                        int translation_mode = int(2*translation_step*1000);

                        float angle_step = 10;
                        int angle_mode = int(2*angle_step*100);

                        for (int iterator = 0; iterator < max_iterator; iterator ++){
                            //******** 2.1 generate Hypothesis point cloud *******//
                            if( parameterDimension == -1){
                                max_iterator = 1;
                            }
                            else if(parameterDimension < 3){
                                para[parameterDimension] += (rand()%translation_mode/1000.0/max_iterator+2*translation_step/max_iterator*iterator-translation_step)*annealing_factor;
                            }
                            else if (parameterDimension < 5){
                                para[parameterDimension] += (rand()%angle_mode/100.0/max_iterator+2*angle_step/max_iterator*iterator-angle_step)*annealing_factor;
                                if(parameterDimension == 4){

                                }
                            }
                            else if (parameterDimension == 5){
                                para[5] += ((para[10]-10)+(para[14]-0)+(para[18]+6.7))/3.0;
                                para[5] += (rand()%angle_mode/100.0/max_iterator+2*angle_step/max_iterator*iterator-angle_step)*annealing_factor;
                            }
                            else if (parameterDimension == 6){
                                para[6] += (rand()%angle_mode/100.0/max_iterator+2*angle_step/max_iterator*iterator-angle_step)*annealing_factor*2;
                            }
                            else if (parameterDimension == 7){
                                para[7] += (rand()%angle_mode/100.0/max_iterator+2*angle_step/max_iterator*iterator-angle_step)*annealing_factor*2;
                            }
                            else if (parameterDimension == 8){
                                para[8] += (rand()%angle_mode/100.0/max_iterator+2*angle_step/max_iterator*iterator-angle_step)*annealing_factor*3;
                            }
                            else if ( parameterDimension == 26){
                                para[26] += (rand()%angle_mode/100.0/max_iterator+2*angle_step/max_iterator*iterator-angle_step)*annealing_factor/2.0;
                            }
                            else if ( (parameterDimension -6)%4 == 0){
                                para[parameterDimension] += (rand()%angle_mode/100.0/max_iterator+2*angle_step/max_iterator*iterator-angle_step)*annealing_factor/2.0;
                            }
                            else{
                                para[parameterDimension] += (rand()%angle_mode/100.0/max_iterator+2*angle_step/max_iterator*iterator-angle_step)*annealing_factor*3.0;
                            }

                            //check parameters:
                            {
                                //min and max
                                for(int i = 0; i < 27; i++){
                                    if(para[i] > openMP_hand.parameters_max[i]){
                                        para[i] = openMP_hand.parameters_max[i];
                                    }
                                    else if (para[i] < openMP_hand.parameters_min[i]){
                                        para[i] = openMP_hand.parameters_min[i];
                                    }
                                }
                                //collision between 2~5 finger
                                for(int i = 10; i < 19; i += 4){
                                    if(fabs(para[i+1]-para[i+5]) < 20){
                                        if(para[i] - para[i+4] < -5){
                                            float dif = para[i+4] - para[i];
                                            para[i+4] -= dif;
                                            para[i] += dif;
                                        }
                                    }
                                }
                            }

                            //generate hypo
                            openMP_hand.set_parameters(para);
                            //ros::Time a = ros::Time::now();
                            openMP_hand.get_joints_positions();
                            //ros::Time time_begin = ros::Time::now();
                            //MyHand.get_handPointCloud(modelPointCloud);
                            openMP_hand.samplePointCloud(openMP_modelPointCloud);
                            //ros::Time time_end = ros::Time::now();
                            //******** 2.1 done ****************//

                            //******** 2.2 Ray tracing for Hypothesis ********//
                            Mat visiblityMap_Hypo(imageSize,imageSize,CV_8UC1,Scalar(back_groud_value));
                           // pcl::PointCloud<pcl::PointXYZRGB> visibleModelPointCloud;
                            //Ray_tracing_OrthognalProjection(modelPointCloud, imageSize, resolution, visiblityMap_Hypo, visibleModelPointCloud);
                            Ray_tracing_OrthognalProjection(openMP_modelPointCloud, imageSize, resolution, visiblityMap_Hypo);
                            //ros::Time time_h = ros::Time::now();
                            //******** 2.2 done *******************//
//                            std::cout <<"Duration 1: " << time_begin - a <<std::endl;
//                            std::cout<<"Duration 2: "<< time_end-time_begin << std::endl;
//                            std::cout<<"Duration 3: "<< time_h-time_end << std::endl;
//                            std::cout<<"Duration a: "<< time_h-a << std::endl;

                            //******** 2.3 Score (similarity assessment) ******//
                            float overlap, overall_diff, overlap_diff, overlap_obs, overlap_hyp;
                            Score(visibilityMap_Oberservation, visiblityMap_Hypo, back_groud_value, overlap, overlap_obs, overlap_hyp, overall_diff, overlap_diff);
                            //******** 2.3 done *************//

                            //std::cout << "Overall_diff: " << overall_diff << std::endl;

                            if((overlap_obs/overlap <= Opt_Score_overlapratio1 || overlap_hyp/overlap <= Opt_Score_overlapratio2) && (overlap_diff/overlap <= Opt_Score_aver_overlapdiff+0.1 || overlap_diff/overlap <= 1.2)){
                                Opt_Score_overlapratio1 = min(overlap_obs/overlap,Opt_Score_overlapratio1);
                                Opt_Score_overlapratio2 = min(overlap_hyp/overlap, Opt_Score_overlapratio2);
                                if(overlap_diff/overlap <= Opt_Score_aver_overlapdiff )
                                    Opt_Score_aver_overlapdiff = 1.0*overlap_diff/overlap;
                                for(int i = 0; i< 27; i++){
                                    optimal_para[i] = para[i];
                                }
                            }

                            for(int i = 0; i< 27; i++){
                                //std::cout << "para" << i <<": " << para[i] << std::endl;
                                para[i] = para_suboptimal[i];
                            }
                        }
                    }
                }

                vector<float> for_the_que;
                for(int i = 0; i< 27; i++){
                    //std::cout << "para" << i <<": " << optimal_para[i] << std::endl;
                    para_lastFrame[i] = optimal_para[i];
                    for_the_que.push_back(optimal_para[i]);
                }
                parameter_que[id].add_new(for_the_que);
                parameter_que[id].smooth_mean(3);
                errors[id].x = Opt_Score_overlapratio1;
                errors[id].y = Opt_Score_overlapratio2;
                errors[id].z = Opt_Score_aver_overlapdiff;

                ros::Time thread_end = ros::Time::now();
               std::cout << id <<"Thread alloc: " << thread_assigned - thread_begin <<std::endl;
               std::cout << id <<"Thread model: " << thread_handModel - thread_assigned <<std::endl;
               std::cout << id <<"Thread total: " << thread_end - thread_begin <<std::endl;

            }

            //Find the best particle out of all
            int best_particle_index = 0, worst_particle_index = 0;
            float weight_a = 2.0, weight_b = 2.0, weight_c = 1.0;
            float best_particle_error = weight_a * errors[0].x + weight_b * errors[0].y + weight_c * errors[0].z;
            float worst_particle_error = weight_a * errors[0].x + weight_b * errors[0].y + weight_c * errors[0].z;
            for(int particle_index = 1; particle_index < PARTICLE_NUM; particle_index++){
                float score = weight_a * errors[particle_index].x+weight_b * errors[particle_index].y+weight_c * errors[particle_index].z;
                if(score<best_particle_error){
                    best_particle_error = score;
                    best_particle_index = particle_index;
                }
                if(score > worst_particle_error){
                    worst_particle_error = score;
                    worst_particle_index = particle_index;
                }
            }
//            std::cout<<"Best particle index: " << best_particle_index << std::endl;
//            std::cout<<"Best particle error: " << errors[best_particle_index].x << ", " <<
//                       errors[best_particle_index].y << ", " << errors[best_particle_index].z <<
//                       " " << best_particle_error<<std::endl;

//            std::cout<<"Worst particle index: " << worst_particle_index << std::endl;
//            std::cout<<"Worst particle error: " << errors[worst_particle_index].x << ", " <<
//                       errors[worst_particle_index].y << ", " << errors[worst_particle_index].z <<
//                       " " << worst_particle_error<<std::endl;

            //reset the worst:
            if(worst_particle_error > 5.8)
            for(int i = 0; i< 27; i++){
                parameter_que[worst_particle_index].para_sequence_smoothed[0][i] = parameter_que[best_particle_index].para_sequence_smoothed[0][i];
            }

            best_parameter_particle.add_new(parameter_que[best_particle_index].para_sequence_smoothed[0]);
            best_parameter_particle.smooth_mean(3);
            for(int i = 0; i< 27; i++){
                det_para[i] = best_parameter_particle.para_delta[i];
            }

            //Calculate best hand/finger pose:
            MyHand.set_parameters(parameter_que[best_particle_index].para_sequence_smoothed[0]);
            MyHand.get_joints_positions();
            MyHand.samplePointCloud(modelPointCloud);

        }



        pcl::PointCloud<pcl::PointXYZRGB> articulation;
        for(int i = 0; i < 27; i++){
            articulation.push_back(MyHand.joints_position[i]);
        }






        //Gaution:
        //        boost::mt19937 *rng = new boost::mt19937();
        //        rng->seed(time(NULL));

        //        boost::normal_distribution<> distribution(70, 10);
        //        boost::variate_generator< boost::mt19937, boost::normal_distribution<> > dist(*rng, distribution);





        seq++;
        if(seq > 10000)
            seq = 1;
        ///////////////////////////////////////////////////////////
        ros::Time Publish_begin = ros::Time::now();

        //ROS_INFO("Prepare Hand Cloud");
        sensor_msgs::PointCloud2 hand_cloud_msg;
        toROSMsg(handcloud,hand_cloud_msg);
        hand_cloud_msg.header.frame_id=hand_kp_pter->header.frame_id;
        hand_cloud_msg.header.stamp = hand_kp_pter->header.stamp;
        handPublisher_.publish(hand_cloud_msg);

        //ROS_INFO("Prepare Articulation");
        sensor_msgs::PointCloud2 cloud_msg_articulation;
        toROSMsg(articulation,cloud_msg_articulation);
        cloud_msg_articulation.header.frame_id=hand_kp_pter->header.frame_id;
        cloud_msg_articulation.header.stamp = hand_kp_pter->header.stamp;
        articulatePublisher_.publish(cloud_msg_articulation);

        //ROS_INFO("Prepare Model Cloud");
        sensor_msgs::PointCloud2 model_cloud_msg;
        toROSMsg(modelPointCloud,model_cloud_msg);
        model_cloud_msg.header.frame_id=hand_kp_pter->header.frame_id;
        model_cloud_msg.header.stamp = hand_kp_pter->header.stamp;
        modelPublisher_.publish(model_cloud_msg);

        //ROS_INFO("Prepare Hand Detection Result");
        visualization_msgs::Marker bone;
        bone.header.frame_id = hand_kp_pter->header.frame_id;
        bone.header.stamp = hand_kp_pter->header.stamp;
        bone.ns = "finger_tracking";
        bone.type = visualization_msgs::Marker::LINE_LIST;
        bone.id = 0;
        bone.action = visualization_msgs::Marker::ADD;
        bone.pose.orientation.w = 1.0;
        bone.scale.x = 0.001;
        bone.color.a = 1.0;
        bone.color.g = 1.0;
        geometry_msgs::Point p2;
        for(int finger = 0; finger <5; finger++){
            for(int i = 1; i< 5; i++){
                p2.x = articulation.points[5*finger+i].x;
                p2.y = articulation.points[5*finger+i].y;
                p2.z = articulation.points[5*finger+i].z;
                bone.points.push_back(p2);
                p2.x = articulation.points[5*finger+i+1].x;
                p2.y = articulation.points[5*finger+i+1].y;
                p2.z = articulation.points[5*finger+i+1].z;
                bone.points.push_back(p2);
            }
        }
        for(int i = 0; i< 2; i++){
            for(int j = 0; j< 3; j++){
                p2.x = articulation.points[6+5*j+i].x;
                p2.y = articulation.points[6+5*j+i].y;
                p2.z = articulation.points[6+5*j+i].z;
                bone.points.push_back(p2);
                p2.x = articulation.points[6+5*j+5+i].x;
                p2.y = articulation.points[6+5*j+5+i].y;
                p2.z = articulation.points[6+5*j+5+i].z;
                bone.points.push_back(p2);
            }
        }
        p2.x = articulation.points[1].x;
        p2.y = articulation.points[1].y;
        p2.z = articulation.points[1].z;
        bone.points.push_back(p2);
        p2.x = articulation.points[6].x;
        p2.y = articulation.points[6].y;
        p2.z = articulation.points[6].z;
        bone.points.push_back(p2);
//        p2.x = articulation.points[1].x;
//        p2.y = articulation.points[1].y;
//        p2.z = articulation.points[1].z;
//        bone.points.push_back(p2);
//        p2.x = articulation.points[21].x;
//        p2.y = articulation.points[21].y;
//        p2.z = articulation.points[21].z;
//        bone.points.push_back(p2);
        bone_pub_.publish( bone );

        //ROS_INFO("Prepare Finger Position from Leap Sensor");
        visualization_msgs::Marker bone_leap;
        bone_leap.header.frame_id = hand_kp_pter->header.frame_id;
        bone_leap.header.stamp = hand_kp_pter->header.stamp;
        bone_leap.ns = "finger_leap";
        bone_leap.type = visualization_msgs::Marker::LINE_LIST;
        bone_leap.id = 0;
        bone_leap.action = visualization_msgs::Marker::ADD;
        bone_leap.pose.orientation.w = 1.0;
        bone_leap.scale.x = 0.001;
        bone_leap.color.a = 1.0;
        bone_leap.color.r = 1.0;
        for(int finger = 0; finger <5; finger++){
            for(int i = 1; i< 5; i++){
                p2.x = hand_kp.points[6*finger+1+i].x;
                p2.y = hand_kp.points[6*finger+1+i].y;
                p2.z = hand_kp.points[6*finger+1+i].z;
                bone_leap.points.push_back(p2);
                p2.x = hand_kp.points[6*finger+i+2].x;
                p2.y = hand_kp.points[6*finger+i+2].y;
                p2.z = hand_kp.points[6*finger+i+2].z;
                bone_leap.points.push_back(p2);
            }
        }
        for(int i = 0; i< 2; i++){
            for(int j = 0; j< 3; j++){
                p2.x = hand_kp.points[8+6*j+i].x;
                p2.y = hand_kp.points[8+6*j+i].y;
                p2.z = hand_kp.points[8+6*j+i].z;
                bone_leap.points.push_back(p2);
                p2.x = hand_kp.points[8+6*j+6+i].x;
                p2.y = hand_kp.points[8+6*j+6+i].y;
                p2.z = hand_kp.points[8+6*j+6+i].z;
                bone_leap.points.push_back(p2);
            }
        }
        p2.x = hand_kp.points[2].x;
        p2.y = hand_kp.points[2].y;
        p2.z = hand_kp.points[2].z;
        bone_leap.points.push_back(p2);
        p2.x = hand_kp.points[8].x;
        p2.y = hand_kp.points[8].y;
        p2.z = hand_kp.points[8].z;
        bone_leap.points.push_back(p2);
        bone_leap_pub_.publish( bone_leap );

        ros::Time Publish_end = ros::Time::now();
        std::cout << Publish_end-Publish_begin << " seconds: Publish messages." << std::endl;










        ros::Time time9 = ros::Time::now();
        std::cout<< time9-time0 << "seconds: Total"<<std::endl;
        ROS_INFO("One callback done");


    }
    catch (std::exception& e)
    {
        //if there is an error during conversion, display it
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

}


