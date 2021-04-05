
/*
 * @Author: Conghao Wong
 * @Date: 2021-01-27 17:46:56
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2021-04-05 22:24:21
 * @Description: file content
 */

using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;

namespace modules.models.Base
{
    public class Dataset
    {
        string _dataset;
        string _dataset_dir;
        List<int> _order;
        List<float> _paras;
        string _video_path;
        List<object> _weights;
        float _scale;

        public string dataset {
            get {
                return this.dataset;
            }
        }
        public string dataset_dir {
            get {
                return this._dataset_dir;
            }
        }
        public List<int> order {
            get {
                return this._order;
            }
        }
        public List<float> paras {
            get {
                return this._paras;
            }
        }
        public string video_path {
            get {
                return this._video_path;
            }
        }
        public List<object> weights {
            get {
                return this._weights;
            }
        }
        public float scale {
            get {
                return this._scale;
            }
        }

        public Dataset(
            string dataset,
            string dataset_dir,
            List<int> order,
            List<float> paras,
            string video_path,
            List<object> weights,
            float scale
        )
        {
            this._dataset = dataset;
            this._dataset_dir = dataset_dir;
            this._order = order;
            this._paras = paras;
            this._video_path = video_path;
            this._weights = weights;
            this._scale = scale;
        }
    }
}