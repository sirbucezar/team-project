using System.Threading.Tasks;
using backend.Models;

namespace backend.Services
{
    public interface IAugmentationService
    {
        Task<bool> ProcessAugmentationAsync(FeedbackAugmentationRequest request);
    }
}