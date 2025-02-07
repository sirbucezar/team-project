using System;
using System.Net;
using System.Threading.Tasks;
using Azure.Storage.Blobs;
using Azure.Storage.Sas;
using Microsoft.Azure.Functions.Worker;
using Microsoft.Azure.Functions.Worker.Http;
using Microsoft.Extensions.Logging;
using backend.Services;
using backend.Helpers;

namespace backend.Functions
{
    public class GetSasFunction
    {
        private readonly ILogger<GetSasFunction> _logger;
        private readonly ISasService _sasService;

        public GetSasFunction(ILogger<GetSasFunction> logger, ISasService sasService)
        {
            _logger = logger;
            _sasService = sasService;
        }

        [Function("GetSasToken")]
        public async Task<HttpResponseData> Run(
            [HttpTrigger(AuthorizationLevel.Function, "get")] HttpRequestData req)
        {
            _logger.LogInformation("GetSas function triggered.");

            var query = System.Web.HttpUtility.ParseQueryString(req.Url.Query);
            var filename = query["filename"];
            if (string.IsNullOrEmpty(filename))
            {
                return await ResponseHelpers.CreateJsonResponseAsync(req, HttpStatusCode.BadRequest, new { error = "Filename parameter is required" });
            }

            try
            {
                // Use the SAS service to generate SAS URL
                string sasUrl = await _sasService.GenerateSasTokenAsync(filename);
                return await ResponseHelpers.CreateJsonResponseAsync(req, HttpStatusCode.OK, new { sas_url = sasUrl });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating SAS token");
                return await ResponseHelpers.CreateJsonResponseAsync(req, HttpStatusCode.InternalServerError, new { error = "Failed to generate SAS token" });
            }
        }
    }
}