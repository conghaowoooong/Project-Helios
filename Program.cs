/*
 * @Author: Conghao Wong
 * @Date: 2021-01-22 19:34:14
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2021-01-28 01:15:28
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
            var dm = new M.Managers.TrainManagers.TrainDataManager(Margs, prepare_type:"all");
        }
    }
}
