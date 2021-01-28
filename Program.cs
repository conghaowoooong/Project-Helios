/*
 * @Author: Conghao Wong
 * @Date: 2021-01-22 19:34:14
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2021-01-29 00:42:58
 * @Description: file content
 */

using M = models;

using NumSharp;
using System;
using System.Linq;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;


namespace csharpmodel
{
    class Program
    {
        static void Main(string[] args)
        {
            var Margs = new M.Managers.ArgManagers.TrainArgsManager();

            dynamic model;
            if (Margs.model == "l"){
                Margs.load = "linear";
                model = new M.Prediction.Linear(Margs);
                model.run_commands();
            }
            
            // var dm = new M.Managers.TrainManagers.TrainDataManager(Margs, prepare_type:"all");
        }
    }
}
