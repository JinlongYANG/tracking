#ifndef tracking_node_hpp
#define tracking_node_hpp

#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <math.h>
#include <ros/ros.h>
#include <dynamic_reconfigure/server.h>
#include <boost/shared_ptr.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CameraInfo.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/TransformStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
//#include <opencv2/core/eigen.hpp>
//#include <opencv2/contrib/contrib.hpp>
#include <image_geometry/pinhole_camera_model.h>
//#include <image_geometry/stereo_camera_model.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
//#include <tf2_ros/transform_broadcaster.h>
//#include <tf2_ros/transform_listener.h>
//#include "tracking/tracking_Config.h"
#include <leap_msgs/Leap.h>
#include <cstdio>

#include "GCoptimization/GCoptimization.h"




using namespace image_transport;
using namespace sensor_msgs;
using namespace geometry_msgs;
using namespace cv;
using namespace image_geometry;
using namespace pcl;
using namespace Eigen;
using namespace std;

class tracking_Node
{
private:
    image_transport::ImageTransport imageTransport_;

    message_filters::TimeSynchronizer<PointCloud2, PointCloud2, Image> timeSynchronizer_;
    message_filters::Subscriber<PointCloud2> hand_kp_Subscriber_;
    message_filters::Subscriber<PointCloud2> hand_Subscriber_;
    message_filters::Subscriber<Image> registered_Depth_Image_Subscriber;

    
    ros::Publisher articulatePublisher_;
    //ros::Publisher leap_articulate_pub_;
    ros::Publisher handPublisher_;
    ros::Publisher segmented_hand_;
    ros::Publisher bone_pub_;
    ros::Publisher modelPublisher_;

public:
    tracking_Node(ros::NodeHandle& nh);
    //void updateConfig(tracking::tracking_Config &config, uint32_t level);

    /////////////////////////////////////////////////////////////////////
    //***************     Call back function     **********************//
    void syncedCallback(const PointCloud2ConstPtr& hand_kp_pter, const PointCloud2ConstPtr& hand_pter, const ImageConstPtr& cvpointer_depthImage);
    /////////////////     end of call back function     /////////////////

    /////////////////////////////////////////////////////////////////////
    //***************     Multi-label Segment     **********************//
    void GridGraphSeqeratePalm(const pcl::PointCloud<pcl::PointXYZRGB> Hand_kp, const pcl::PointCloud<pcl::PointXYZRGB> Hand_point_cloud, pcl::PointCloud<pcl::PointXYZRGB> & labeled_hand_kp, vector<int> &label){

        int m_width = Hand_point_cloud.size();
        int m_height = 1;
        int m_num_pixels = m_width * m_height;
        int m_num_labels = 16;
        int *m_result = new int[m_num_pixels];   // stores result of optimization


        //prepare the joints data: 0~7 are 8 joints on palm; 8~11 are 4 joints of thumb;
        //12~15 are 4 joints of index finger; 16~19 middle finger; 20~23 ring finger; 24~27 pinky finger
        pcl::PointCloud<pcl::PointXYZRGB> Hand_joints;
        //palm: (8 9 14 15 20 21 26 27)
        for(int i = 8; i<27;i=i+6){
            Hand_joints.push_back(Hand_kp.points[i]);
            Hand_joints.push_back(Hand_kp.points[i+1]);
        }
        //fingers: thumb 3,4,5,6; index 9,10,11,12; middle 15,16,17,18; ring 21,22,23,24; pinky 27,28,29,30
        for(int i = 3; i<28; i = i+6){
            for(int j = 0; j<4; j++){
                Hand_joints.push_back(Hand_kp.points[i+j]);
            }
        }

        // first set up the array for data costs
        int *m_data = new int[m_num_pixels*m_num_labels];
        for (size_t i = 0; i < Hand_point_cloud.points.size (); ++i){

            //l==0: palm
            float a = 0, b = 0, c = 0;
            //index finger matacarpal; middle finger matacarpal; ring finger matacarpal; pinky finger matacarpal
            float temp_min = 255;
            for ( int f = 0; f < 4; f++){
                int pre = 2*f;
                int next = pre + 1;
                a = sqrt((Hand_joints.points[pre].x - Hand_joints.points[next].x)*(Hand_joints.points[pre].x - Hand_joints.points[next].x)
                         +(Hand_joints.points[pre].y - Hand_joints.points[next].y) * (Hand_joints.points[pre].y - Hand_joints.points[next].y)
                         +(Hand_joints.points[pre].z - Hand_joints.points[next].z) * (Hand_joints.points[pre].z - Hand_joints.points[next].z));
                b = sqrt((Hand_joints.points[pre].x - Hand_point_cloud.points[i].x)*(Hand_joints.points[pre].x - Hand_point_cloud.points[i].x)
                         + (Hand_joints.points[pre].y - Hand_point_cloud.points[i].y)*(Hand_joints.points[pre].y - Hand_point_cloud.points[i].y)
                         + (Hand_joints.points[pre].z - Hand_point_cloud.points[i].z)*(Hand_joints.points[pre].z - Hand_point_cloud.points[i].z));
                c = sqrt((Hand_joints.points[next].x - Hand_point_cloud.points[i].x)*(Hand_joints.points[next].x - Hand_point_cloud.points[i].x)
                         + (Hand_joints.points[next].y - Hand_point_cloud.points[i].y)*(Hand_joints.points[next].y - Hand_point_cloud.points[i].y)
                         + (Hand_joints.points[next].z - Hand_point_cloud.points[i].z)*(Hand_joints.points[next].z - Hand_point_cloud.points[i].z));
                //if point projectiong is out of link; else on link
                if( a*a + min(b,c)*min(b,c)< max(b,c)*max(b,c)){
                    if(temp_min > min(b,c))
                        temp_min = min(b,c);
                }
                else{
                    if ( temp_min > 0.5*sqrt((a+b+c)*(a+b-c)*(b+c-a)*(a-b+c))/a )
                        temp_min = 0.5*sqrt((a+b+c)*(a+b-c)*(b+c-a)*(a-b+c))/a;
                }
            }
            m_data[i * m_num_labels] = 1000.0*temp_min;


            //l==2~17: fingers
            //l==2: thumb metacarpal; l==3: thumb proximal; l==4: thumb distal
            //l==5: index finger proximal; l==6: index finger intermediate; l==7: index finger distal
            //l==8: middle finger proximal; l==9: middle finger intermediate; l==10: middle finger distal
            //l==11: ring finger proximal; l==12: index ring intermediate; l==13: ring finger distal
            //l==14: pinky finger proximal; l==15: pinky finger intermediate; l==16: pinky finger distal
            for ( int f = 0; f < 5; f++){
                for ( int k = 0; k< 3; k++){
                    int pre = 4*f+k+8;
                    int next = pre+1;
                    int l = f*3+k+1;
                    a = sqrt((Hand_joints.points[pre].x - Hand_joints.points[next].x)*(Hand_joints.points[pre].x - Hand_joints.points[next].x)
                             +(Hand_joints.points[pre].y - Hand_joints.points[next].y) * (Hand_joints.points[pre].y - Hand_joints.points[next].y)
                             +(Hand_joints.points[pre].z - Hand_joints.points[next].z) * (Hand_joints.points[pre].z - Hand_joints.points[next].z));
                    b = sqrt((Hand_joints.points[pre].x - Hand_point_cloud.points[i].x)*(Hand_joints.points[pre].x - Hand_point_cloud.points[i].x)
                             + (Hand_joints.points[pre].y - Hand_point_cloud.points[i].y)*(Hand_joints.points[pre].y - Hand_point_cloud.points[i].y)
                             + (Hand_joints.points[pre].z - Hand_point_cloud.points[i].z)*(Hand_joints.points[pre].z - Hand_point_cloud.points[i].z));
                    c = sqrt((Hand_joints.points[next].x - Hand_point_cloud.points[i].x)*(Hand_joints.points[next].x - Hand_point_cloud.points[i].x)
                             + (Hand_joints.points[next].y - Hand_point_cloud.points[i].y)*(Hand_joints.points[next].y - Hand_point_cloud.points[i].y)
                             + (Hand_joints.points[next].z - Hand_point_cloud.points[i].z)*(Hand_joints.points[next].z - Hand_point_cloud.points[i].z));
                    //if point projectiong is out of link; else on link
                    if( a*a + min(b,c)*min(b,c)< max(b,c)*max(b,c)){
                        m_data[i * m_num_labels + l] = 1000.0*min(b,c);
                    }
                    else{
                        m_data[i * m_num_labels + l] = 1000.0*0.5*sqrt((a+b+c)*(a+b-c)*(b+c-a)*(a-b+c))/a;
                    }
                }
            }
        }


        // next set up the array for smooth costs
        int *smooth = new int[m_num_labels*m_num_labels];
        for ( int l1 = 0; l1 < m_num_labels; l1++ ){
            for (int l2 = 0; l2 < m_num_labels; l2++ ){
                //                if(l1 == 0 && l2%3 == 1)
                //                    smooth[l1+l2*m_num_labels] = 1;
                //                else if (l1%3 == 1 && l2%3 == 2)
                //                    smooth[l1+l2*m_num_labels] = 1;
                //                else if (l1%3 == 2 && l2%3 == 0 && !l2)
                //                    smooth[l1+l2*m_num_labels] = 1;

                //                else if(l2 == 0 && l1%3 == 1)
                //                    smooth[l1+l2*m_num_labels] = 1;
                //                else if (l2%3 == 1 && l1%3 == 2)
                //                    smooth[l1+l2*m_num_labels] = 1;
                //                else if (l2%3 == 2 && l1%3 == 0 && !l1)
                //                    smooth[l1+l2*m_num_labels] = 1;

                //                else
                smooth[l1+l2*m_num_labels] = 10;
            }
        }




        try{
            GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(m_num_pixels,m_num_labels);


            gc->setDataCost(m_data);
            gc->setSmoothCost(smooth);

            // now set up neighborhood system
            // first build Kdtree:
            pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr ptr (new pcl::PointCloud<pcl::PointXYZRGB> (Hand_point_cloud));
            kdtree.setInputCloud (ptr);

            // find K nearest neighbours for each point, and set neighbours:
            for (size_t i = 0; i < ptr->points.size (); ++i){
                pcl::PointXYZRGB searchPoint = ptr->points[i];
                int K = 8;

                std::vector<int> pointIdxNKNSearch(K);
                std::vector<float> pointNKNSquaredDistance(K);

                //                std::cout << "K nearest neighbor search at (" << searchPoint.x
                //                          << " " << searchPoint.y
                //                          << " " << searchPoint.z
                //                          << ") with K=" << K << std::endl;

                if ( kdtree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
                {
                    for (size_t j = 0; j < pointIdxNKNSearch.size (); ++j){
                        //                        std::cout << "    "  <<   ptr->points[ pointIdxNKNSearch[j] ].x
                        //                                  << " " << ptr->points[ pointIdxNKNSearch[j] ].y
                        //                                  << " " << ptr->points[ pointIdxNKNSearch[j] ].z
                        //                                  << " (squared distance: " << pointNKNSquaredDistance[j] << ")" << std::endl;
                        gc->setNeighbors(i,pointIdxNKNSearch[j]);
                    }
                }

            }

            printf("Before optimization energy is %lld\n",gc->compute_energy());
            //            for ( int  i = 0; i < m_num_pixels; i++ ){
            //                m_result[i] = gc->whatLabel(i);
            //                std::cout<<m_result[i]<<" ";
            //                if((i+1)%10 == 0)
            //                    std::cout<<std::endl;
            //            }
            //gc->expansion(2);// run expansion for 1 iterations. For swap use gc->swap(num_iterations);
            gc->swap(5);
            printf("After optimization energy is %lld\n",gc->compute_energy());

            //            for ( int  i = 0; i < m_num_pixels; i++ ){
            //                m_result[i] = gc->whatLabel(i);
            //                std::cout<<m_result[i]<<" ";
            //                if((i+1)%10 == 0)
            //                    std::cout<<std::endl;
            //            }


            //            for ( int row = 0; row < Hand_depth.rows; row++){
            //                for ( int col = 0; col < Hand_depth.cols; col++){
            //                    int label = gc->whatLabel(row * Hand_depth.cols + col);
            //                    if(label == 0){
            //                        output.at<unsigned char>(row, 3*col+0) = 255;
            //                        output.at<unsigned char>(row, 3*col+1) = 255;
            //                        output.at<unsigned char>(row, 3*col+2) = 255;
            //                    }
            //                    else if(label%3 == 0)
            //                        output.at<unsigned char>(row, 3*col+0) = label*15;
            //                    if(label%3 == 1)
            //                        output.at<unsigned char>(row, 3*col+1) = label*15;
            //                    if(label%3 == 2)
            //                        output.at<unsigned char>(row, 3*col+2) = label*15;

            //                }
            //            }
            for (size_t i = 0; i < Hand_point_cloud.points.size (); ++i){
                int pointlabel = gc->whatLabel(i);
                PointXYZRGB p = Hand_point_cloud.points[i];
                uint8_t r = 0,g = 0,b = 0;
                if(pointlabel == 0){
                    r = 200;
                    g = 200;
                    b = 200;
                }
                else if(pointlabel%3 == 0){
                    r = pointlabel*13+60;
                }
                else if(pointlabel%3 == 1){

                    g = pointlabel*13+60;
                }
                else if(pointlabel%3 == 2){
                    b = pointlabel*13+60;
                }

                uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
                p.rgb = *reinterpret_cast<float*>(&rgb);
                labeled_hand_kp.push_back(p);
                label.push_back(pointlabel);
            }

            delete gc;
        }
        catch (GCException e){
            e.Report();
        }

        delete [] m_result;
        delete [] smooth;
        delete [] m_data;
        //cv::waitKey();
    }
    /////////////////     end of Multi-label Segment     /////////////////

    /////////////////////////////////////////////////////////////////////
    //***************    Nearest Neighbour Segment     **********************//
    void NearestNeighbour(const pcl::PointCloud<pcl::PointXYZRGB> Hand_kp, const pcl::PointCloud<pcl::PointXYZRGB> Hand_point_cloud, pcl::PointCloud<pcl::PointXYZRGB> & labeled_hand_kp, vector<int> &label){
        //prepare the joints data: 0~7 are 8 joints on palm; 8~11 are 4 joints of thumb;
        //12~15 are 4 joints of index finger; 16~19 middle finger; 20~23 ring finger; 24~27 pinky finger
        pcl::PointCloud<pcl::PointXYZRGB> Hand_joints;
        //palm: (8 9 14 15 20 21 26 27)
        for(int i = 8; i<27;i=i+6){
            Hand_joints.push_back(Hand_kp.points[i]);
            Hand_joints.push_back(Hand_kp.points[i+1]);
        }
        //fingers: thumb 3,4,5,6; index 9,10,11,12; middle 15,16,17,18; ring 21,22,23,24; pinky 27,28,29,30
        for(int i = 3; i<28; i = i+6){
            for(int j = 0; j<4; j++){
                Hand_joints.push_back(Hand_kp.points[i+j]);
            }
        }

        for (size_t i = 0; i < Hand_point_cloud.points.size (); ++i){
            float temp = 255;
            int temp_l = 255;

            //l==0: palm
            float a = 0, b = 0, c = 0;
            //index finger matacarpal; middle finger matacarpal; ring finger matacarpal; pinky finger matacarpal
            float temp_min = 255;
            for ( int f = 0; f < 4; f++){
                int pre = 2*f;
                int next = pre + 1;
                a = sqrt((Hand_joints.points[pre].x - Hand_joints.points[next].x)*(Hand_joints.points[pre].x - Hand_joints.points[next].x)
                         +(Hand_joints.points[pre].y - Hand_joints.points[next].y) * (Hand_joints.points[pre].y - Hand_joints.points[next].y)
                         +(Hand_joints.points[pre].z - Hand_joints.points[next].z) * (Hand_joints.points[pre].z - Hand_joints.points[next].z));
                b = sqrt((Hand_joints.points[pre].x - Hand_point_cloud.points[i].x)*(Hand_joints.points[pre].x - Hand_point_cloud.points[i].x)
                         + (Hand_joints.points[pre].y - Hand_point_cloud.points[i].y)*(Hand_joints.points[pre].y - Hand_point_cloud.points[i].y)
                         + (Hand_joints.points[pre].z - Hand_point_cloud.points[i].z)*(Hand_joints.points[pre].z - Hand_point_cloud.points[i].z));
                c = sqrt((Hand_joints.points[next].x - Hand_point_cloud.points[i].x)*(Hand_joints.points[next].x - Hand_point_cloud.points[i].x)
                         + (Hand_joints.points[next].y - Hand_point_cloud.points[i].y)*(Hand_joints.points[next].y - Hand_point_cloud.points[i].y)
                         + (Hand_joints.points[next].z - Hand_point_cloud.points[i].z)*(Hand_joints.points[next].z - Hand_point_cloud.points[i].z));
                //if point projectiong is out of link; else on link
                if( a*a + min(b,c)*min(b,c)< max(b,c)*max(b,c)){
                    if(temp_min > min(b,c))
                        temp_min = min(b,c);
                }
                else{
                    if ( temp_min > 0.5*sqrt((a+b+c)*(a+b-c)*(b+c-a)*(a-b+c))/a )
                        temp_min = 0.5*sqrt((a+b+c)*(a+b-c)*(b+c-a)*(a-b+c))/a;
                }
            }
            if ( temp > temp_min ){
                temp = temp_min;
                temp_l = 0;
            }


            //l==2~17: fingers
            //l==2: thumb metacarpal; l==3: thumb proximal; l==4: thumb distal
            //l==5: index finger proximal; l==6: index finger intermediate; l==7: index finger distal
            //l==8: middle finger proximal; l==9: middle finger intermediate; l==10: middle finger distal
            //l==11: ring finger proximal; l==12: index ring intermediate; l==13: ring finger distal
            //l==14: pinky finger proximal; l==15: pinky finger intermediate; l==16: pinky finger distal
            for ( int f = 0; f < 5; f++){
                for ( int k = 0; k< 3; k++){
                    int pre = 4*f+k+8;
                    int next = pre+1;
                    int l = f*3+k+1;
                    a = sqrt((Hand_joints.points[pre].x - Hand_joints.points[next].x)*(Hand_joints.points[pre].x - Hand_joints.points[next].x)
                             +(Hand_joints.points[pre].y - Hand_joints.points[next].y) * (Hand_joints.points[pre].y - Hand_joints.points[next].y)
                             +(Hand_joints.points[pre].z - Hand_joints.points[next].z) * (Hand_joints.points[pre].z - Hand_joints.points[next].z));
                    b = sqrt((Hand_joints.points[pre].x - Hand_point_cloud.points[i].x)*(Hand_joints.points[pre].x - Hand_point_cloud.points[i].x)
                             + (Hand_joints.points[pre].y - Hand_point_cloud.points[i].y)*(Hand_joints.points[pre].y - Hand_point_cloud.points[i].y)
                             + (Hand_joints.points[pre].z - Hand_point_cloud.points[i].z)*(Hand_joints.points[pre].z - Hand_point_cloud.points[i].z));
                    c = sqrt((Hand_joints.points[next].x - Hand_point_cloud.points[i].x)*(Hand_joints.points[next].x - Hand_point_cloud.points[i].x)
                             + (Hand_joints.points[next].y - Hand_point_cloud.points[i].y)*(Hand_joints.points[next].y - Hand_point_cloud.points[i].y)
                             + (Hand_joints.points[next].z - Hand_point_cloud.points[i].z)*(Hand_joints.points[next].z - Hand_point_cloud.points[i].z));
                    //if point projectiong is out of link; else on link
                    if( a*a + min(b,c)*min(b,c)< max(b,c)*max(b,c)){
                        if ( temp > min(b,c) ){
                            temp = min(b,c);
                            temp_l = l;
                        }
                    }
                    else{
                        if ( temp > 0.5*sqrt((a+b+c)*(a+b-c)*(b+c-a)*(a-b+c))/a ){
                            temp = 0.5*sqrt((a+b+c)*(a+b-c)*(b+c-a)*(a-b+c))/a;
                            temp_l = l;
                        }
                    }
                }
            }

            PointXYZRGB p = Hand_point_cloud.points[i];
            uint8_t r = temp_l*16+15;
            uint8_t g = 255-(temp_l*16+15);
            uint8_t blue = 0;
            if (temp_l%2 == 0)
                blue = 128;
            uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)blue);
            p.rgb = *reinterpret_cast<float*>(&rgb);
            labeled_hand_kp.push_back(p);
            label.push_back(temp_l);


        }
    }
    /////////////////     end of Nearest Neighbour Segment     /////////////////

    /////////////////////////////////////////////////////////////////////
    //***************    Ray tracing -> orthognal projection     **********************//
    void Ray_tracing_OrthognalProjection(const pcl::PointCloud<pcl::PointXYZRGB> cloud, const int imageSize, const int resolution, Mat &visibilityMap){
        //std::cout<<"Cloud size: " << cloud.points.size() << " Image size: " << imageSize << std::endl;
        if(imageSize*imageSize/2 > cloud.points.size()){
            ROS_ERROR("Too high resolution for ray tracing");
            int x, y, z, row, col, depth;
            for ( size_t i = 0; i < cloud.points.size(); ++i ){
                x = int(cloud.points[i].x * 1000);
                y = int(cloud.points[i].y * 1000);
                z = int(cloud.points[i].z * 1000);
                row = y/resolution + imageSize/2;
                col = x/resolution + imageSize/2;
                depth = z/resolution + imageSize/2;
                if( row < imageSize && col < imageSize){
                    visibilityMap.at<unsigned char>(row, col) = min(int(visibilityMap.at<unsigned char>(row, col)), depth);
                }
            }
        }
        else{
            int x, y, z, row, col, depth;
            for ( size_t i = 0; i < cloud.points.size(); ++i ){
                x = int(cloud.points[i].x * 1000);
                y = int(cloud.points[i].y * 1000);
                z = int(cloud.points[i].z * 1000);
                row = y/resolution + imageSize/2;
                col = x/resolution + imageSize/2;
                depth = z/resolution + imageSize/2;
                if( row < imageSize && col < imageSize){
                    visibilityMap.at<unsigned char>(row, col) = min(int(visibilityMap.at<unsigned char>(row, col)), depth);
                }
            }
        }
    }

    void Ray_tracing_OrthognalProjection(const pcl::PointCloud<pcl::PointXYZRGB> cloud, const int imageSize, const int resolution, Mat &visibilityMap, pcl::PointCloud<pcl::PointXYZRGB> &visiblecloud){
        //std::cout<<"Cloud size: " << cloud.points.size() << "Image size: " << imageSize << std::endl;
        if(imageSize*imageSize/2 > cloud.points.size()){
            ROS_ERROR("Too high resolution for ray tracing");
            return;
        }
        else{
            int x, y, z, row, col, depth;
            for ( size_t i = 0; i < cloud.points.size(); ++i ){
                x = int(cloud.points[i].x * 1000);
                y = int(cloud.points[i].y * 1000);
                z = int(cloud.points[i].z * 1000);
                row = y/resolution + imageSize/2;
                col = x/resolution + imageSize/2;
                depth = z/resolution + imageSize/2;
                if( row < imageSize && col < imageSize){
                    if( row < imageSize && col < imageSize){
                        visibilityMap.at<unsigned char>(row, col) = min(int(visibilityMap.at<unsigned char>(row, col)), depth);
                    }
                }
            }
            pcl::PointXYZRGB p;
            uint8_t r = 255, g= 255, b= 255;
            uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
            p.rgb = *reinterpret_cast<float*>(&rgb);

            for ( row = 0; row < visibilityMap.rows; ++row){
                for ( col = 0; col < visibilityMap.cols; ++col){
                    if(visibilityMap.at<unsigned char>(row, col)!=100){
                        p.x = (col-imageSize/2)*resolution/1000.0;
                        p.y = (row-imageSize/2)*resolution/1000.0;
                        p.z = (visibilityMap.at<unsigned char>(row, col)-imageSize/2)*resolution/1000.0;
                        visiblecloud.push_back(p);
                    }
                }
            }
        }
    }
    /////////////////     end of Ray tracing -> orthognal projection     /////////////////

    /////////////////////////////////////////////////////////////////////
    //***************    Score naive subtraction     **********************//
    void Score(const Mat Ober, const Mat Hypo, float &score){
        score = 0;
        for( int row = 0; row < Ober.rows; ++row){
            for( int col = 0; col < Ober.cols; ++col){
                score+= abs(Ober.at<unsigned char>(row, col) - Hypo.at<unsigned char>(row, col));
            }
        }

    }

    void Score(const Mat Ober, const Mat Hypo, const int backgroud_value, float &overlap, float &overall_diff, float &overlap_diff){
        overlap = 0;
        overall_diff = 0;
        overlap_diff = 0;
        for( int row = 0; row < Ober.rows; ++row){
            for( int col = 0; col < Ober.cols; ++col){
                overall_diff += abs(Ober.at<unsigned char>(row, col) - Hypo.at<unsigned char>(row, col));
                if( Ober.at<unsigned char>(row, col) != backgroud_value && Hypo.at<unsigned char>(row, col) != backgroud_value){
                    overlap++;
                    overlap_diff += abs(Ober.at<unsigned char>(row, col) - Hypo.at<unsigned char>(row, col));
                }
            }
        }

    }

    void Score(const Mat Ober, const Mat Hypo, const int backgroud_value, float &overlap, float &overlap_obs, float &overlap_hyp, float &overall_diff, float &overlap_diff){
        overlap = 0;
        overall_diff = 0;
        overlap_diff = 0;
        overlap_obs = 0;
        overlap_hyp = 0;
        for( int row = 0; row < Ober.rows; ++row){
            for( int col = 0; col < Ober.cols; ++col){
                overall_diff += abs(Ober.at<unsigned char>(row, col) - Hypo.at<unsigned char>(row, col));
                if( Ober.at<unsigned char>(row, col) != backgroud_value){
                    overlap_obs++;
                    if(Hypo.at<unsigned char>(row, col) != backgroud_value){
                        overlap++;
                        overlap_diff += abs(Ober.at<unsigned char>(row, col) - Hypo.at<unsigned char>(row, col));
                    }
                }
                if(Hypo.at<unsigned char>(row, col) != backgroud_value){
                    overlap_hyp++;
                }
            }
        }
    }
    /////////////////     end of Score (similarity assessment)     /////////////////


};
#endif
