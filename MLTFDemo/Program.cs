using System;
using System.IO;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace MLTFDemo
{
    class Program
    {
        public const int FeatureLength = 600;
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "sentiment_model");

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            var lookupMap = mlContext.Data.LoadFromTextFile(Path.Combine(_modelPath, "imdb_word_index.csv"),
                columns: new[]
                   {
                        new TextLoader.Column("Words", DataKind.String, 0),
                        new TextLoader.Column("Ids", DataKind.Int32, 1),
                   },
                separatorChar: ','
               );

            Action<VariableLength, FixedLength> ResizeFeaturesAction = (s, f) =>
            {
                var features = s.VariableLengthFeatures;
                Array.Resize(ref features, FeatureLength);
                f.Features = features;
            };

            TensorFlowModel tensorFlowModel = mlContext.Model.LoadTensorFlowModel(_modelPath);

            DataViewSchema schema = tensorFlowModel.GetModelSchema();
            Console.WriteLine(" =============== TensorFlow Model Schema =============== ");
            var featuresType = (VectorDataViewType)schema["Features"].Type;
            Console.WriteLine($"Name: Features, Type: {featuresType.ItemType.RawType}, Size: ({featuresType.Dimensions[0]})");
            var predictionType = (VectorDataViewType)schema["Prediction/Softmax"].Type;
            Console.WriteLine($"Name: Prediction/Softmax, Type: {predictionType.ItemType.RawType}, Size: ({predictionType.Dimensions[0]})");


            IEstimator<ITransformer> pipeline =
                mlContext.Transforms.Text.TokenizeIntoWords("TokenizedWords", "ReviewText")
                .Append(mlContext.Transforms.Conversion.MapValue("VariableLengthFeatures", lookupMap,
                    lookupMap.Schema["Words"], lookupMap.Schema["Ids"], "TokenizedWords"))
                .Append(mlContext.Transforms.CustomMapping(ResizeFeaturesAction, "Resize"))
                .Append(tensorFlowModel.ScoreTensorFlowModel("Prediction/Softmax", "Features"))
                .Append(mlContext.Transforms.CopyColumns("Prediction", "Prediction/Softmax"));

            IDataView dataView = mlContext.Data.LoadFromEnumerable(new List<MovieReview>());
            ITransformer model = pipeline.Fit(dataView);

            PredictSentiment(mlContext, model);
        }

        public static void PredictSentiment(MLContext mlContext, ITransformer model)
        {
            var engine = mlContext.Model.CreatePredictionEngine<MovieReview, MovieReviewSentimentPrediction>(model);

            var review = new MovieReview()
            {
                ReviewText = "this film is really good"
            };

            var sentimentPrediction = engine.Predict(review);

            Console.WriteLine("Number of classes: {0}", sentimentPrediction.Prediction.Length);
            Console.WriteLine("Is sentiment/review positive? {0}", sentimentPrediction.Prediction[1] > 0.5 ? "Yes." : "No.");

            /////////////////////////////////// Expected output ///////////////////////////////////
            //
            // Name: Features, Type: System.Int32, Size: 600
            // Name: Prediction/Softmax, Type: System.Single, Size: 2
            //
            // Number of classes: 2
            // Is sentiment/review positive ? Yes
            // Prediction Confidence: 0.65
        }

        public class MovieReview
        {
            public string ReviewText { get; set; }
        }

        public class MovieReviewSentimentPrediction
        {
            [VectorType(2)]
            public float[] Prediction { get; set; }
        }

        public class VariableLength
        {
            [VectorType]
            public int[] VariableLengthFeatures { get; set; }
        }

        public class FixedLength
        {
            [VectorType(FeatureLength)]
            public int[] Features { get; set; }
        }
    }
}
