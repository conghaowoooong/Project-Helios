/*
 * @Author: Conghao Wong
 * @Date: 2021-01-26 13:17:07
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2021-01-28 18:47:09
 * @Description: file content
 */

using System;
using System.IO;
using System.Text;
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
using OpenCvSharp;
using System.Runtime.Serialization.Formatters.Binary;

namespace models
{
    public static class HelpMethods
    {
        public static void log_function(string str, string end = "\n")
        {
            var output = String.Format("{0}{1}", str, end);
            Console.Write(output);
        }

        public static void log_percent(int current, int total, int log_step = 10){
            
        }

        public static string get_slice_index(int start, int end)
        {
            string slice_format = "{0}:{1}";
            return string.Format(slice_format, start, end);
        }

        public static NDArray where1d(NDArray condition)
        {
            List<int> all_index = new List<int>();

            for (int index = 0; index < len(condition); index++)
            {
                bool cond = condition[index];
                if (cond)
                {
                    all_index.append(index);
                }
            }

            var all_index_numpy = np.array(all_index);
            return all_index_numpy;
        }

        public static string dir_check(string path)
        {
            if (System.IO.Directory.Exists(path) == false)
            {
                System.IO.Directory.CreateDirectory(path);
            }
            return path;
        }

        public static bool dir_exist(string path)
        {
            return System.IO.Directory.Exists(path);
        }

        public static bool file_exist(string path){
            return System.IO.File.Exists(path);
        }

        public static void write_file(string path, dynamic data)
        {
            var fs = new FileStream(path, FileMode.Create);
            var bf = new BinaryFormatter();
            bf.Serialize(fs, data);
            fs.Close();
        }

        public static T read_file<T>(string path)
        {
            var fs = new FileStream(path, FileMode.Open);
            var bf = new BinaryFormatter();
            T ps = (T)bf.Deserialize(fs);
            return ps;
        }

        public static dynamic load_image(string path, int[] resize_shape = null, bool return_numpy = true){
            var image = tf.io.read_file(path);
            var image_decode = tf.image.decode_jpeg(image);
            if (!(resize_shape == null)){
                image_decode = tf.image.resize(image_decode, resize_shape);
            }
            if (return_numpy){
                var result = image_decode.numpy().reshape(image_decode.shape);
                return result;
            }
            return image_decode;
        }

        public static NDArray csv2ndarray(string path)
        {
            var data_array = CSVTool.Read(path, Encoding.Default);

            NDArray data_ndarray = np.zeros(new int[] { len(data_array), len(data_array[0]) });
            foreach (var index in range(len(data_array)))
            {
                for (int index_j = 0; index_j < len(data_array[index]); index_j++)
                {
                    data_ndarray[index, index_j] = Convert.ToSingle(data_array[index][index_j]);
                }
            }
            return data_ndarray;
        }

        public static NDArray mat2ndarray(Mat frame)
        {
            var channels = frame.Channels();
            var width = frame.Size().Width;
            var height = frame.Size().Height;

            byte[] data = new byte[width * height * channels];
            frame.GetArray<byte>(out data); //获取mat数据到（byte）data

            NDArray nd = new NDArray(data); //直接创建一个NDarray读取data
            return nd;
        }

        public static Dictionary<T1, T2> make_dict<T1, T2>(T1[] keys, T2[] values)
        {
            Dictionary<T1, T2> output = new Dictionary<T1, T2>();
            var length = len(keys);
            foreach (var index in range(length))
            {
                output.Add(keys[index], values[index]);
            }
            return output;
        }

        private static (NDArray Y_p, NDArray B) _predict_linear(
            NDArray x, NDArray y, NDArray x_p, float diff_weights = 0
        )
        {
            Tensor Pt;
            if (diff_weights == 0)
            {
                Pt = tf.diag(tf.ones((x.shape[0]), tf.float32));
            }
            else
            {
                Pt = tf.diag(tf.nn.softmax(
                    tf.pow(tf.cast(tf.range(1, 1 + x.shape[0]), tf.float32), diff_weights)
                ));
            }
            var P = Pt.numpy();
            var A = np.stack(new NDArray[] { np.ones_like(x), x }).T;
            var A_p = np.stack(new NDArray[] { np.ones_like(x_p), x_p }).T;
            var Y = y.T;
            var B = np.matmul(np.matmul(np.matmul(ndarray_inv(np.matmul(np.matmul(A.T, P), A)), A.T), P), Y);
            var Y_p = np.matmul(A_p, B);

            return (Y_p, B);
        }

        public static NDArray predict_linear_for_person(NDArray position, int time_pred, float different_weights = 0.95f)
        {
            var time_obv = position.shape[0];
            var t = np.arange(time_obv);
            var t_p = np.arange(time_pred);
            var x = position.T[0];
            var y = position.T[1];

            (var x_p, var _) = _predict_linear(t, x, t_p, diff_weights: different_weights);
            (var y_p, var _) = _predict_linear(t, y, t_p, diff_weights: different_weights);

            return np.concatenate(new NDArray[] { x_p, y_p }, axis:-1);
        }

        public static NDArray ndarray_inv(NDArray A)
        {
            var n = A.shape[0];
            if (n != A.shape[1])
            {
                return null;
            }

            NDArray C = np.zeros((n, 2 * n));
            NDArray D = np.zeros((n, n));

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    C[i, j] = A[i, j];
                }
            }
            for (int i = 0; i < n; i++)
            {
                C[i, i + n] = 1;
            }
            for (int k = 0; k < n; k++)
            {
                double max = Math.Abs(C[k, k]);
                int ii = k;
                for (int m = k + 1; m < n; m++)
                    if (max < Math.Abs(C[m, k]))
                    {
                        max = Math.Abs(C[m, k]);
                        ii = m;
                    }
                for (int m = k; m < 2 * n; m++)
                {
                    if (ii == k) break;
                    double c;
                    c = C[k, m];
                    C[k, m] = C[ii, m];
                    C[ii, m] = c;
                }
                if ((bool)!(C[k, k] == 1))
                {
                    double bs = C[k, k];
                    if (bs == 0)
                    {
                        // Console.WriteLine("求逆错误！结果可能不正确！");
                        break;
                        //return null;
                    }
                    C[k, k] = 1;
                    for (int p = k + 1; p < n * 2; p++)
                    {
                        C[k, p] /= bs;
                    }
                }
                for (int q = k + 1; q < n; q++)
                {
                    double bs = C[q, k];
                    for (int p = k; p < n * 2; p++)
                    {
                        C[q, p] -= bs * C[k, p];
                    }
                }
            }
            for (int q = n - 1; q > 0; q--)
            {
                for (int k = q - 1; k > -1; k--)
                {
                    double bs = C[k, q];
                    for (int m = k + 1; m < 2 * n; m++)
                    {
                        C[k, m] -= bs * C[q, m];
                    }
                }
            }
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    D[i, j] = C[i, j + n];
            return D;
        }

        public static (double ade, double fde) calculate_ADE_FDE_numpy(NDArray pred, NDArray GT)
        {
            double ade, fde;
            if (len(pred.shape) == 3)
            {    // [K, pred, 2]
                var ade_list = new List<double>();
                var fde_list = new List<double>();
                for (int index = 0; index < len(pred); index ++)
                {   
                    var p = pred[index];
                    var all_loss = np.mean(np.sqrt(np.square(p - GT)), axis: 1);
                    ade_list.append(np.mean(all_loss));
                    fde_list.append(all_loss[-1]);
                }

                var min_index = np.argmin(np.array(ade_list));
                ade = ade_list[min_index];
                fde = fde_list[min_index];

                // # # ADE of the mean traj
                // # mean_traj = np.mean(pred, axis=0)
                // # mean_traj_loss = np.linalg.norm(mean_traj - GT, ord=2, axis=1)
                // # ade = np.mean(mean_traj_loss)
                // # fde = mean_traj_loss[-1]

            }
            else
            {
                var all_loss = np.mean(np.sqrt(np.square(pred - GT)), axis: 1);
                ade = np.mean(all_loss);
                fde = all_loss[-1];
            }

            return (ade, fde);
        }

        public static NDArray linear_activation(NDArray x, float a = 1.0f, float b = 1.0f){
            var zero = np.zeros_like(x, dtype:np.float32);
            return (x < zero) * a * x + (x > zero) * b * x + (x == zero) * 1.0f * x;
        }

        public static NDArray calculate_length(NDArray vec){
            return np.sqrt(np.sum(np.square(vec).astype(np.float32))).astype(np.float32);
        }

        public static NDArray calculate_cosine(NDArray vec1, NDArray vec2){
            var length1 = calculate_length(vec1);
            var length2 = calculate_length(vec2);

            if ((bool)(length2 == 0)){
                return np.array(-1.0f).astype(np.float32);
            } else {
                return (np.sum(vec1 * vec2) / (length1 * length2)).astype(np.float32);
            }
        }
    }

    public static class CSVTool
    {
        private static char _csvSeparator = ',';
        private static bool _trimColumns = false;
        //获取一个单元格的写入格式
        public static string GetCSVFormat(string str)
        {
            string tempStr = str;
            if (str.Contains(","))
            {
                if (str.Contains("\""))
                {
                    tempStr = str.Replace("\"", "\"\"");
                }
                tempStr = "\"" + tempStr + "\"";
            }
            return tempStr;
        }
        //获取一行的写入格式
        public static string GetCSVFormatLine(List<string> strList)
        {
            string tempStr = "";
            for (int i = 0; i < strList.Count - 1; i++)
            {
                string str = strList[i];
                tempStr = tempStr + GetCSVFormat(str) + ",";
            }
            tempStr = tempStr + GetCSVFormat(strList[strList.Count - 1]) + "\n";
            return tempStr;
        }
        //解析一行
        public static List<string> ParseLine(string line)
        {
            StringBuilder _columnBuilder = new StringBuilder();
            List<string> Fields = new List<string>();
            bool inColum = false;//是否是在一个列元素里
            bool inQuotes = false;//是否需要转义
            bool isNotEnd = false;//读取完毕未结束转义
            _columnBuilder.Remove(0, _columnBuilder.Length);

            //空行也是一个空元素，一个逗号是2个空元素
            if (line == "")
            {
                Fields.Add("");
            }
            // Iterate through every character in the line  遍历行中的每个字符
            for (int i = 0; i < line.Length; i++)
            {
                char character = line[i];

                //If we are not currently inside a column   如果我们现在不在一列中
                if (!inColum)
                {
                    // If the current character is a double quote then the column value is contained within
                    //如果当前字符是双引号，则列值包含在内
                    // double quotes, otherwise append the next character
                    //双引号，否则追加下一个字符
                    inColum = true;
                    if (character == '"')
                    {
                        inQuotes = true;
                        continue;
                    }
                }
                // If we are in between double quotes   如果我们处在双引号之间
                if (inQuotes)
                {
                    if ((i + 1) == line.Length)//这个字符已经结束了整行
                    {
                        if (character == '"')//正常转义结束，且该行已经结束
                        {
                            inQuotes = false;
                            continue;
                        }
                        else//异常结束，转义未收尾
                        {
                            isNotEnd = true;
                        }
                    }
                    else if (character == '"' && line[i + 1] == _csvSeparator)//结束转义，且后面有可能还有数据
                    {
                        inQuotes = false;
                        inColum = false;
                        i++;//跳过下一个字符
                    }
                    else if (character == '"' && line[i + 1] == '"')//双引号转义
                    {
                        i++;//跳过下一个字符
                    }
                    else if (character == '"')//双引号单独出现（这种情况实际上已经是格式错误，为了兼容暂时不处理）
                    {
                        throw new System.Exception("格式错误，错误的双引号转义");
                    }
                    //其他情况直接跳出，后面正常添加
                }
                else if (character == _csvSeparator)
                {
                    inColum = false;
                }
                // If we are no longer in the column clear the builder and add the columns to the list
                ////结束该元素时inColumn置为false，并且不处理当前字符，直接进行Add
                if (!inColum)
                {
                    Fields.Add(_trimColumns ? _columnBuilder.ToString().Trim() : _columnBuilder.ToString());
                    _columnBuilder.Remove(0, _columnBuilder.Length);
                }
                else//追加当前列
                {
                    _columnBuilder.Append(character);
                }
            }

            // If we are still inside a column add a new one （标准格式一行结尾不需要逗号结尾，而上面for是遇到逗号才添加的，为了兼容最后还要添加一次）
            if (inColum)
            {
                if (isNotEnd)
                {
                    _columnBuilder.Append("\n");
                }
                Fields.Add(_trimColumns ? _columnBuilder.ToString().Trim() : _columnBuilder.ToString());
            }
            else  //如果inColumn为false，说明已经添加，因为最后一个字符为分隔符，所以后面要加上一个空元素
            {
                Fields.Add("");
            }
            return Fields;
        }
        //读取文件
        public static List<List<string>> Read(string filePath, Encoding encoding)
        {
            List<List<string>> result = new List<List<string>>();
            string content = File.ReadAllText(filePath, encoding);
            string[] lines = content.Split(new string[] { "\n" }, StringSplitOptions.RemoveEmptyEntries);
            for (int i = 0; i < lines.Length; i++)
            {
                List<string> line = ParseLine(lines[i]);
                result.Add(line);
            }
            return result;
        }
        //写入文件
        public static void Write(string filePath, Encoding encoding, List<List<string>> result)
        {
            StringBuilder builder = new StringBuilder();
            for (int i = 0; i < result.Count; i++)
            {
                List<string> line = result[i];
                builder.Append(GetCSVFormatLine(line));
            }
            File.WriteAllText(filePath, builder.ToString(), encoding);
        }
        //打印
        public static void Debug(List<List<string>> result)
        {
            for (int i = 0; i < result.Count; i++)
            {
                List<string> line = result[i];
                for (int j = 0; j < line.Count; j++)
                {
                    // UnityEngine.Debug.LogWarning(line[j]);
                }
            }
        }

        public static string get_value(List<List<string>> result, int i, int j)
        {
            List<string> line = result[i];
            return line[j];
        }
    }
}