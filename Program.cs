﻿/*
 * @Author: Conghao Wong
 * @Date: 2021-01-22 19:34:14
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2021-01-29 02:23:02
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


namespace ProjectHelios
{
    class Program
    {
        static void Main(string[] args)
        {
            var Margs = new M.Managers.ArgManagers.TrainArgsManager();
            Margs.model = "bgm";
            dynamic model;
            if (Margs.model == "l")
            {
                Margs.load = "linear";
                model = new M.Prediction.Linear(Margs);
                model.run_commands();
            }
            else if ( Margs.model == "bgm")
            {
                model = new M.Prediction.NewBGM(Margs);
                model.run_commands();
            }

            // var dm = new M.Managers.TrainManagers.TrainDataManager(Margs, prepare_type:"all");
        }
    }
}
