/*
 * @Author: Conghao Wong
 * @Date: 2021-01-22 21:08:04
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2021-04-05 21:52:11
 * @Description: file content
 */

using System;
using modules.models.Base;

namespace modules.models.Prediction
{
    public class BasePredictArgs : BaseArgs
    {
        public int obs_frames = 8;
        public int pred_frames = 12;
    }

    public class TrainArgs : BasePredictArgs
    {
        public int map_half_size = 50;

        // train args
        // public int batch_size = 5000;
        public float lr = 0.001f;
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


    public class OnlineArgs : TrainArgs
    {
        public int wait_frames = 4;
        public int guidance_map_limit = 10000;
        public (int, int) order = (0, 1);
        
        public int draw_future = 0;
        public string vis = "show";
        public string img_save_base_path = "./online_vis";

        public int focus_mode = 0;
        public int run_frames = 0;
    }
}

