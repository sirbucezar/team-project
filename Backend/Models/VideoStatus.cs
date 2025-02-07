using System;

namespace backend.Models
{
    public class VideoStatus
    {
        public string PartitionKey { get; set; }
        public string RowKey { get; set; }
        public string Status { get; set; }
        public string Message { get; set; }
        public DateTime Timestamp { get; set; }
    }
}