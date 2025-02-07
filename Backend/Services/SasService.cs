using System;
using System.Net;
using System.Threading.Tasks;
using Azure.Storage.Blobs;
using Azure.Storage.Sas;
using Microsoft.Extensions.Logging;

namespace backend.Services
{
    public class SasService : ISasService
    {
        private readonly ILogger<SasService> _logger;
        private readonly string _connectionString;

        public SasService(ILogger<SasService> logger)
        {
            _logger = logger;
            _connectionString = Environment.GetEnvironmentVariable("AzureWebJobsStorage") 
                ?? throw new InvalidOperationException("AzureWebJobsStorage is not configured");
        }

        public async Task<string> GenerateSasTokenAsync(string filename)
        {
            var blobServiceClient = new BlobServiceClient(_connectionString);
            var containerClient = blobServiceClient.GetBlobContainerClient("uploads");
            await containerClient.CreateIfNotExistsAsync(Azure.Storage.Blobs.Models.PublicAccessType.None);

            var blobClient = containerClient.GetBlobClient(filename);

            var sasBuilder = new BlobSasBuilder
            {
                BlobContainerName = containerClient.Name,
                BlobName = blobClient.Name,
                Resource = "b",
                StartsOn = DateTimeOffset.UtcNow,
                ExpiresOn = DateTimeOffset.UtcNow.AddMinutes(30)
            };
            sasBuilder.SetPermissions(BlobSasPermissions.Read | BlobSasPermissions.Write);

            try
            {
                var sasUri = blobClient.GenerateSasUri(sasBuilder);
                return sasUri.ToString();
            }
            catch (InvalidOperationException)
            {
                var accountName = blobServiceClient.AccountName;
                var accountKey = Environment.GetEnvironmentVariable("StorageAccountKey") 
                    ?? throw new InvalidOperationException("StorageAccountKey is not configured");
                var storageSharedKeyCredential = new Azure.Storage.StorageSharedKeyCredential(accountName, accountKey);
                var sasQueryParameters = sasBuilder.ToSasQueryParameters(storageSharedKeyCredential);
                return $"{blobClient.Uri}?{sasQueryParameters}";
            }
        }
    }
}