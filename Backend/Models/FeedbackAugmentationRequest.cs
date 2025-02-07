using System;
using System.Text.Json.Serialization;
using System.Collections.Generic;


namespace backend.Models
{
    public class FeedbackAugmentationRequest
    {
        [JsonPropertyName("processing_id")]
        public string ProcessingId { get; set; }

        [JsonPropertyName("exercise")]
        public string Exercise { get; set; }

        [JsonPropertyName("user_id")]
        public string UserId { get; set; }

        [JsonPropertyName("stageAnalysis")]
        public Dictionary<string, StageAnalysis> StageAnalysis { get; set; }
    }

    public class StageAnalysis
    {
        [JsonPropertyName("video")]
        public string Video { get; set; }

        [JsonPropertyName("classified_score")]
        public float ClassifiedScore { get; set; }

        [JsonPropertyName("predicted_score")]
        public float PredictedScore { get; set; }

        [JsonPropertyName("video_url")]
        public string VideoUrl { get; set; }
    }
}
// Path: Backend/Models/FeedbackAugmentationResponse.cs