using System.Text.Json.Serialization;

namespace backend.Models
{
    public class StageInfo
    {
        [JsonPropertyName("name")]
        public string Name { get; set; }

        [JsonPropertyName("start_time")]
        public float StartTime { get; set; }

        [JsonPropertyName("end_time")]
        public float EndTime { get; set; }
    }
}