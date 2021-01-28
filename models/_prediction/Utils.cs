/*
 * @Author: Conghao Wong
 * @Date: 2021-01-26 19:17:12
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2021-01-28 00:50:05
 * @Description: file content
 */

using System;
using System.Collections.Generic;
using NumSharp;
using System.Linq;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using static models.HelpMethods;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using models.Managers;
using models.Managers.AgentManagers;

namespace models.Prediction{
    public static class Utils{
        public static (List<Tensor> model_inputs, Tensor gt) getInputs_onlyTraj(List<BaseAgentManager> input_agents){
            List<List<Tensor>> model_inputs = new List<List<Tensor>>();
            List<Tensor> gt = new List<Tensor>();

            model_inputs.append(new List<Tensor>());
            foreach (var agent in input_agents){
                model_inputs[0].append(agent.get_traj());
                gt.append(agent.get_future_traj());
            }

            List<Tensor> tensor_inputs = new List<Tensor>();
            foreach (var index in range(len(model_inputs))){
                tensor_inputs[index] = tf.cast(tf.stack(model_inputs[index]), tf.float32);
            }

            var gt_input = tf.cast(tf.stack(gt), tf.float32);
            return (tensor_inputs, gt_input);
        }
    }
}