/*
 * @Author: Conghao Wong
 * @Date: 2021-01-22 21:08:04
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2021-04-05 21:15:49
 * @Description: file content
 */

using System;

namespace modules.models.Base
{
    public class BaseArgs
    {
        public string gpu = "0";
        public bool verbose = false;

        public string save_base_dir = "./logs";
        public string save_format = "tf";
        public string log_dir = "null";
        public string load = "null";
        public int batch_size = 5000;
    }
}

