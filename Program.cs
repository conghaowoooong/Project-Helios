/*
 * @Author: Conghao Wong
 * @Date: 2021-01-22 19:34:14
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2021-04-07 09:59:49
 * @Description: file content
 */

using NumSharp;
using System;
using System.Linq;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

using mod = modules;


namespace ProjectHelios
{
    class Program
    {
        static void Main(string[] args)
        {
            var Margs = new mod.models.Prediction.TrainArgs();
            Margs.model = "test";
            Margs.load = "./logs/sa_K8_zara1";
            Margs.K_train = 8;
            
            dynamic model;
            if (Margs.model == "l")
            {
                Margs.load = "linear";
                model = new mod.Linear.Linear(Margs);
                model.run_commands();
            }
            else if (Margs.model == "test"){
                model = new mod.satoshi.SatoshiAlpha(Margs);
                model.run_commands();
            }

            // var dm = new M.Managers.TrainManagers.TrainDataManager(Margs, prepare_type:"all");
        }
    }
}
