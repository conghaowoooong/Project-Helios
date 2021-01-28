/*
 * @Author: Conghao Wong
 * @Date: 2021-01-22 21:08:04
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2021-01-29 01:47:45
 * @Description: file content
 */

using System;

namespace models.Managers.ArgManagers
{
    public class BaseArgManager{
        public string gpu = "0";
        public bool verbose = false;

        public string save_base_dir = "./logs";
        public string save_format = "tf";
        public string log_dir = "null";
        public string load = "null";        
    }

    public class BasePredictArgs : BaseArgManager{
        public int obs_frames = 8;
        public int pred_frames = 12;
    }

    public class TrainArgsManager : BasePredictArgs{
        public int map_half_size = 50;
        
        public int batch_size = 5000;
        public string test_mode = "one";

        // dataset base settings
        public string dataset = "ethucy";
        public string test_set = "zara1";
        public string force_set = "null";

        // social args
        public int init_position = 10000;
        public float window_size_expand_meter = 10.0f;
        public int window_size_guidance_map = 10;
        public int avoid_size = 15;
        public int interest_size = 20;

        // dataset training settings
        public int step = 4;
        public int add_noise = 0;
        public int rotate = 0;

        // linear args
        public float diff_weights = 0.95f;

        // prediction model args
        public string model = "l";
    }
}

