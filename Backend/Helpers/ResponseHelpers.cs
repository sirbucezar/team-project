using System.Net;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Azure.Functions.Worker;
using Microsoft.Azure.Functions.Worker.Http;

namespace backend.Helpers
{
    public static class ResponseHelpers
    {
        public static async Task<HttpResponseData> CreateJsonResponseAsync(HttpRequestData req, HttpStatusCode statusCode, object content)
        {
            var response = req.CreateResponse(statusCode);
            response.Headers.Add("Content-Type", "application/json");
            await response.WriteStringAsync(JsonSerializer.Serialize(content));
            return response;
        }
    }
}