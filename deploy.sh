#!/bin/bash

# Landmark-based Drowsiness Detection System Deployment Script
# ============================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
    
    # Check system resources
    print_status "Checking system resources for landmark detection..."

    # Check available memory
    if command -v free &> /dev/null; then
        available_mem=$(free -m | awk 'NR==2{printf "%.0f", $7}')
        if [ "$available_mem" -lt 2048 ]; then
            print_warning "Low available memory ($available_mem MB). Recommend at least 2GB for stable operation."
        else
            print_success "Sufficient memory available ($available_mem MB)"
        fi
    fi
    
    print_success "Prerequisites check completed"
}

# Function to create necessary directories
create_directories() {
    print_status "Creating necessary directories..."

    mkdir -p logs
    mkdir -p data

    # Set permissions
    chmod 755 logs data

    print_success "Directories created"
}

# Function to check available ports
check_ports() {
    print_status "Checking port availability..."

    PORT=8003
    if netstat -tuln 2>/dev/null | grep -q ":$PORT "; then
        print_error "Port $PORT is already in use. Please stop the service using this port."
        exit 1
    fi

    print_success "Port $PORT is available"
}

# Function to build and start the landmark service
deploy_services() {
    print_status "Building and deploying landmark service..."

    docker-compose up -d

    print_success "Landmark service deployed successfully"
}

# Function to wait for service to be ready
wait_for_services() {
    print_status "Waiting for landmark service to be ready..."

    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:8003/ &> /dev/null; then
            print_success "Landmark service is healthy and ready"
            return 0
        fi

        print_status "Attempt $attempt/$max_attempts - Waiting for service..."
        sleep 10
        ((attempt++))
    done

    print_warning "Service may not be fully ready yet. Check logs for details."
}

# Function to show service status
show_status() {
    print_status "Service Status:"
    docker-compose ps

    echo ""
    print_status "Service URL:"
    echo "  Landmark API: http://localhost:8003"

    echo ""
    print_status "API Endpoints:"
    echo "  Process Video: http://localhost:8003/api/process"
    echo "  Get Results: http://localhost:8003/api/results"
    echo "  Webhook Management: http://localhost:8003/api/webhook"

    echo ""
    print_status "To view logs:"
    echo "  docker-compose logs -f"

    echo ""
    print_status "To stop service:"
    echo "  docker-compose down"
}

# Function to show help
show_help() {
    echo "Landmark-based Drowsiness Detection System Deployment Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  -h, --help     Show this help message"
    echo "  -s, --status   Show service status"
    echo "  -c, --clean    Clean up containers and volumes"
    echo "  --rebuild      Rebuild containers from scratch"
    echo ""
    echo "Examples:"
    echo "  $0                    # Deploy landmark service"
    echo "  $0 --status          # Show service status"
    echo "  $0 --clean           # Clean up everything"
    echo "  $0 --rebuild         # Rebuild from scratch"
}

# Function to clean up
cleanup() {
    print_status "Cleaning up containers and volumes..."
    docker-compose down -v
    docker system prune -f
    print_success "Cleanup completed"
}

# Function to rebuild
rebuild() {
    print_status "Rebuilding containers from scratch..."
    docker-compose down
    docker-compose build --no-cache
    print_success "Rebuild completed"
}

# Main script logic
main() {
    local show_status_only=false
    local clean_only=false
    local rebuild_only=false

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -s|--status)
                show_status_only=true
                shift
                ;;
            -c|--clean)
                clean_only=true
                shift
                ;;
            --rebuild)
                rebuild_only=true
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Execute based on options
    if [ "$show_status_only" = true ]; then
        show_status
        exit 0
    fi
    
    if [ "$clean_only" = true ]; then
        cleanup
        exit 0
    fi
    
    if [ "$rebuild_only" = true ]; then
        rebuild
        exit 0
    fi
    
    # Main deployment flow
    print_status "Starting Landmark-based Drowsiness Detection System deployment..."

    check_prerequisites
    create_directories
    check_ports
    deploy_services
    wait_for_services
    show_status

    print_success "Deployment completed successfully!"
}

# Run main function with all arguments
main "$@"
