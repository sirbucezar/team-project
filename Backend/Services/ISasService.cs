using System.Threading.Tasks;

namespace backend.Services
{
    public interface ISasService
    {
        Task<string> GenerateSasTokenAsync(string filename);
    }
}