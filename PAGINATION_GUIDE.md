# Pagination Implementation Guide

This document provides a comprehensive guide to the pagination functionality implemented for the Drowsiness Detection API.

## Overview

The pagination system provides efficient data browsing capabilities for both Evidence Results and Processing Queue data through enhanced API endpoints and user-friendly UI components.

## API Endpoints

### 1. Evidence Results API (`/api/results`)

**Endpoint:** `GET /api/results`

**Parameters:**
- `page` (integer, optional): Page number (default: 1)
- `per_page` (integer, optional): Items per page (default: 10, max: 100)

**Example Request:**
```
GET /api/results?page=2&per_page=25
```

**Response Format:**
```json
{
  "success": true,
  "data": [
    {
      "id": 1,
      "video_url": "http://example.com/video.mp4",
      "process_time": 2.45,
      "yawn_frames": 5,
      "eye_closed_frames": 12,
      "normal_state_frames": 150,
      "is_drowsy": true,
      "processing_status": "processed",
      "created_at": "2024-01-15T10:30:00Z"
    }
  ],
  "pagination": {
    "current_page": 2,
    "per_page": 25,
    "total_pages": 10,
    "total_items": 250,
    "has_next": true,
    "has_previous": true,
    "next_page": 3,
    "previous_page": 1
  }
}
```

### 2. Processing Queue API (`/api/queue`)

**Endpoint:** `GET /api/queue`

**Parameters:**
- `page` (integer, optional): Page number - when provided, returns paginated queue items
- `per_page` (integer, optional): Items per page (default: 10, max: 100)

**Note:** When `page` parameter is not provided, the endpoint returns queue statistics (original functionality).

**Example Request:**
```
GET /api/queue?page=1&per_page=10
```

**Response Format:**
```json
{
  "success": true,
  "data": [
    {
      "id": 1,
      "video_url": "http://example.com/video.mp4",
      "status": "pending",
      "created_at": "2024-01-15T10:30:00Z"
    }
  ],
  "pagination": {
    "current_page": 1,
    "per_page": 10,
    "total_pages": 5,
    "total_items": 50,
    "has_next": true,
    "has_previous": false,
    "next_page": 2,
    "previous_page": null
  }
}
```

## UI Components

### 1. Standalone Pages

#### Paginated Results (`/paginated-results`)
- Clean, responsive interface for browsing evidence results
- Items per page selector (10, 25, 50, 100)
- Previous/Next navigation with page numbers
- Loading states and error handling
- Real-time pagination information display

#### Paginated Queue (`/paginated-queue`)
- Similar interface for processing queue items
- Status badges for different queue states
- Action buttons for checking individual item status
- Responsive design matching the existing dashboard

#### Pagination Demo (`/pagination-demo`)
- Comprehensive demonstration of both APIs
- Tabbed interface showing both data types
- Live API information and examples
- Perfect for testing and understanding the functionality

### 2. Reusable Component

#### Pagination Component (`/templates/components/pagination.html`)
A reusable JavaScript class that can be integrated into any page:

```javascript
// Initialize pagination component
const pagination = initializePaginationComponent({
    apiUrl: '/api/results',
    onDataReceived: function(data) {
        // Handle received data
        displayData(data);
    },
    onError: function(error) {
        // Handle errors
        console.error('Pagination error:', error);
    },
    additionalParams: {
        // Any additional parameters to send with requests
        filter: 'active'
    }
});

// Update parameters dynamically
pagination.setAdditionalParams({ status: 'processed' });

// Get current state
const currentPage = pagination.getCurrentPage();
const totalItems = pagination.getTotalItems();
```

## Features

### Backend Features
- **Efficient Pagination:** Uses LIMIT and OFFSET for database queries
- **Parameter Validation:** Ensures page and per_page values are within acceptable ranges
- **Comprehensive Metadata:** Returns all necessary pagination information
- **Backward Compatibility:** Maintains existing API functionality
- **Error Handling:** Robust error handling with meaningful messages

### Frontend Features
- **Responsive Design:** Works on desktop and mobile devices
- **Loading States:** Visual feedback during data loading
- **Error Handling:** User-friendly error messages
- **Smooth Navigation:** Intuitive pagination controls
- **Customizable:** Easy to modify items per page
- **Real-time Info:** Shows current page and total items
- **Clean Styling:** Matches existing dashboard design

## Integration Examples

### Basic AJAX Integration
```javascript
function loadPaginatedData(page = 1, perPage = 10) {
    $.ajax({
        url: '/api/results',
        method: 'GET',
        data: { page: page, per_page: perPage },
        success: function(response) {
            if (response.success) {
                displayData(response.data);
                updatePagination(response.pagination);
            }
        },
        error: function(xhr, status, error) {
            console.error('Error loading data:', error);
        }
    });
}
```

### Using with Filters
```javascript
function loadFilteredData(filters, page = 1) {
    const params = {
        page: page,
        per_page: 25,
        ...filters
    };
    
    $.ajax({
        url: '/api/results',
        data: params,
        success: function(response) {
            // Handle response
        }
    });
}
```

## Database Methods

The following new methods were added to the database manager:

### Evidence Results
- `get_evidence_results_count()`: Returns total count of evidence results
- `get_all_evidence_results(limit, offset)`: Enhanced with proper pagination

### Processing Queue
- `get_queue_items_count()`: Returns total count of queue items
- `get_all_queue_items(limit, offset)`: New method for paginated queue data

## Performance Considerations

- **Database Indexing:** Ensure proper indexes on `created_at` columns for efficient sorting
- **Limit Per Page:** Maximum of 100 items per page to prevent performance issues
- **Caching:** Consider implementing caching for frequently accessed pages
- **Connection Pooling:** Use connection pooling for high-traffic scenarios

## Testing

### Manual Testing
1. Visit `/pagination-demo` to test both APIs interactively
2. Try different page sizes and navigate through pages
3. Test error scenarios (invalid page numbers, network issues)
4. Verify responsive design on different screen sizes

### API Testing
```bash
# Test evidence results pagination
curl "http://localhost:8002/api/results?page=1&per_page=10"

# Test queue pagination
curl "http://localhost:8002/api/queue?page=1&per_page=5"

# Test queue statistics (original functionality)
curl "http://localhost:8002/api/queue"
```

## Troubleshooting

### Common Issues
1. **Empty Results:** Check if data exists in the database
2. **Slow Loading:** Verify database indexes and query performance
3. **UI Not Updating:** Check browser console for JavaScript errors
4. **Pagination Not Working:** Ensure proper API response format

### Debug Tips
- Use browser developer tools to inspect network requests
- Check server logs for database errors
- Verify pagination metadata in API responses
- Test with different data volumes

## Future Enhancements

- **Search Integration:** Add search functionality to pagination
- **Sorting Options:** Allow sorting by different columns
- **Export Features:** Add export functionality for paginated data
- **Real-time Updates:** Implement WebSocket updates for live data
- **Advanced Filters:** More sophisticated filtering options
- **Bookmarkable URLs:** URL-based pagination state management

## Conclusion

This pagination implementation provides a robust, scalable solution for browsing large datasets in the Drowsiness Detection system. The combination of efficient backend APIs and user-friendly frontend components ensures excellent performance and user experience.

For questions or issues, refer to the demo page at `/pagination-demo` or check the implementation in `simplify.py` and the template files.
