using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace backend.Models
{
    public class VideoProcessMessage
    {
        [JsonPropertyName("processing_id")]
        public string ProcessingId { get; set; }

        [JsonPropertyName("exercise")]
        public string Exercise { get; set; }

        [JsonPropertyName("video_url")]
        public string VideoUrl { get; set; }

        [JsonPropertyName("stages")]
        public List<StageInfo> Stages { get; set; }

        [JsonPropertyName("user_id")]
        public string UserId { get; set; }

        [JsonPropertyName("deployment_id")]
        public string DeploymentId { get; set; }

        [JsonPropertyName("timestamp")]
        public DateTimeOffset Timestamp { get; set; }
    }
}