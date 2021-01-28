/*
 * @Author: Conghao Wong
 * @Date: 2021-01-22 21:08:04
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2021-01-28 15:57:23
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
    }
}

