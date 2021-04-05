/*
 * @Author: Conghao Wong
 * @Date: 2021-01-28 18:50:30
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2021-04-05 20:24:49
 * @Description: file content
 */

using NumSharp;
using System.Collections.Generic;
using System;

using static modules.models.helpMethods.HelpMethods;

namespace modules.models.Base
{
    public class LogFunction
    {
        bool v = true;
        List<string> stringg = new List<string>();
        int max_line = 500;

        public void write(string s, string end = "\n")
        {

        }

        public class LogBar
        {
            private int log_step;
            private NDArray record;

            public LogBar(int log_step = 10)
            {
                this.log_step = log_step;
                this.record = np.zeros((log_step)).astype(np.int32);
            }

            public void log(int current, int total)
            {
                float percent = (float)current * 100 / (float)total;
                int stage = (int)percent / this.log_step;

                if ((int)this.record[stage] == 0)
                {
                    log_function(String.Format("{0}%", this.log_step * stage), end: ".. ");
                    this.record[stage] = 1;
                }

                if (current == total - 1)
                {
                    log_function("Done.");
                }
            }

            public void clean()
            {
                this.record = np.zeros((log_step)).astype(np.int32);
            }
        }
    }
}