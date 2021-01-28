/*
 * @Author: Conghao Wong
 * @Date: 2021-01-27 17:46:56
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2021-01-28 00:36:54
 * @Description: file content
 */

using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;

namespace models.Managers.DatasetManagers{
    public class Dataset{
        public string dataset;
        public string dataset_dir;
        public List<int> order;
        public List<float> paras;
        public string video_path;
        public List<object> weights;
        public float scale;
        public Dataset(
            string dataset,
            string dataset_dir,
            List<int> order,
            List<float> paras,
            string video_path,
            List<object> weights,
            float scale
        ){
            this.dataset = dataset;
            this.dataset_dir = dataset_dir;
            this.order = order;
            this.paras = paras;
            this.video_path = video_path;
            this.weights = weights;
            this.scale = scale;
        }
    }


    public class PredictionDatasetManager{
        Dictionary<string, Dataset> datasets = new Dictionary<string, Dataset>();
        public Dictionary<string, List<string>> dataset_list = new Dictionary<string, List<string>>();
        public List<string> ethucy_testsets;

        public PredictionDatasetManager(){
            this.datasets["eth"] = new Dataset(
                dataset     :   "eth",
                dataset_dir :   "./data/eth/univ",
                order       :   new List<int> {1, 0},
                paras       :   new List<float> {6, 25},
                video_path  :   "./videos/eth.avi",
                weights     :   new List<object> {new double[,] {
                                    {2.8128700e-02, 2.0091900e-03, -4.6693600e+00},
                                    {8.0625700e-04, 2.5195500e-02, -5.0608800e+00},
                                    {3.4555400e-04, 9.2512200e-05, 4.6255300e-01},
                                }, 0.65, 225, 0.6, 160},
                scale       :   1
            );

            this.datasets["hotel"] = new Dataset(
                dataset     :   "hotel",
                dataset_dir :   "./data/eth/hotel",
                order       :   new List<int> {0, 1},
                paras       :   new List<float> {10, 25},
                video_path  :   "./videos/hotel.avi",
                weights     :   new List<object> {new double[,]{
                                    {-1.5966000e-03, 1.1632400e-02, -5.3951400e+00},
                                    {1.1048200e-02, 6.6958900e-04, -3.3295300e+00},
                                    {1.1190700e-04, 1.3617400e-05, 5.4276600e-01},
                                }, 0.54, 470, 0.54, 300},
                scale       :   1
            );

            this.datasets["zara1"] = new Dataset(
                dataset     :   "zara1",
                dataset_dir :   "./data/ucy/zara/zara01",
                order       :   new List<int> {1, 0},
                paras       :   new List<float> {10, 25},
                video_path  :   "./videos/zara1.mp4",
                weights     :   new List<object> {-42.54748107, 580.5664891, 47.29369894, 3.196071003},
                scale       :   1
            );

            this.datasets["zara2"] = new Dataset(
                dataset     :   "zara2",
                dataset_dir :   "./data/ucy/zara/zara02",
                order       :   new List<int> {1, 0},
                paras       :   new List<float> {10, 25},
                video_path  :   "./videos/zara2.avi",
                weights     :   new List<object> {-42.54748107, 630.5664891, 47.29369894, 3.196071003},
                scale       :   1
            );

            this.datasets["univ"] = new Dataset(
                dataset     :   "univ",
                dataset_dir :   "./data/ucy/univ/students001",
                order       :   new List<int> {1, 0},
                paras       :   new List<float> {10, 25},
                video_path  :   "./videos/students003.avi",
                weights     :   new List<object> {-41.1428, 576, 48, 0},
                scale       :   1
            );

            this.datasets["zara3"] = new Dataset(
                dataset     :   "zara3",
                dataset_dir :   "./data/ucy/zara/zara03",
                order       :   new List<int> {1, 0},
                paras       :   new List<float> {10, 25},
                video_path  :   "./videos/zara2.avi",
                weights     :   new List<object> {-42.54748107, 630.5664891, 47.29369894, 3.196071003},
                scale       :   1
            );

            this.datasets["univ3"] = new Dataset(
                dataset     :   "univ3",
                dataset_dir :   "./data/ucy/univ/students003",
                order       :   new List<int> {1, 0},
                paras       :   new List<float> {10, 25},
                video_path  :   "./videos/students003.avi",
                weights     :   new List<object> {-41.1428, 576, 48, 0},
                scale       :   1
            );

            this.datasets["unive"] = new Dataset(
                dataset     :   "unive",
                dataset_dir :   "./data/ucy/univ/uni_examples",
                order       :   new List<int> {1, 0},
                paras       :   new List<float> {10, 25},
                video_path  :   "./videos/students003.avi",
                weights     :   new List<object> {-41.1428, 576, 48, 0},
                scale       :   1
            );

            this.ethucy_testsets = new List<string> {"eth", "hotel", "zara1", "zara2", "univ"};
            this.dataset_list["ethucy"] = new List<string> {"eth", "hotel", "zara1", "zara2", "univ", "zara3", "univ3", "unive"};
            this.dataset_list["ethucytest"] = new List<string> {"eth", "hotel", "zara1", "zara2", "univ"};
        }

        public Dataset call(string dataset){
            return this.datasets[dataset];
        }
    }
}