using System.Net;
using System.Threading.Tasks;
using Microsoft.Azure.Functions.Worker;
using Microsoft.Azure.Functions.Worker.Http;
using Microsoft.Extensions.Logging;
using backend.Models;
using backend.Helpers;

namespace backend.Functions
{
    public class VideoStatusFunction
    {
        private readonly ILogger<VideoStatusFunction> _logger;

        public VideoStatusFunction(ILogger<VideoStatusFunction> logger)
        {
            _logger = logger;
        }

        [Function("GetVideoStatus")]
        public async Task<HttpResponseData> Run(
            [HttpTrigger(AuthorizationLevel.Function, "get", Route = "status/{processingId}")] HttpRequestData req,
            [TableInput("VideoProcessing", "{processingId}", "status")] VideoStatus statusEntity,
            string processingId)
        {
            _logger.LogInformation($"Getting status for processing ID: {processingId}");

            if (statusEntity == null)
            {
                return await ResponseHelpers.CreateJsonResponseAsync(req, HttpStatusCode.NotFound, new { error = $"No status found for processing ID: {processingId}" });
            }

            return await ResponseHelpers.CreateJsonResponseAsync(req, HttpStatusCode.OK, new
            {
                processingId = processingId,
                status = statusEntity.Status,
                message = statusEntity.Message,
                timestamp = statusEntity.Timestamp
            });
        }
    }
}