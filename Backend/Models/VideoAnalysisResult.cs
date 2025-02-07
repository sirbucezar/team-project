using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace backend.Models
{
    public class VideoAnalysisResult
    {
        [JsonPropertyName("processingId")]
        public string ProcessingId { get; set; }

        [JsonPropertyName("stage_analysis")]
        public Dictionary<string, object> StageAnalysis { get; set; }

        [JsonPropertyName("warnings")]
        public List<string> Warnings { get; set; }

        [JsonPropertyName("metrics")]
        public Dictionary<string, double> Metrics { get; set; }
    }
}