const API_BASE_URL = 'http://localhost:8000';

export interface Video {
  filename: string;
  path: string;
  size: number;
  created_time: string;
  modified_time: string;
  duration: string;
  location: string;
  camera: string;
}

export interface PersonCount {
  known_persons: number;
  unknown_persons: number;
  total_detections: number;
  unique_known_persons: number;
  unique_unknown_persons: number;
  last_updated: string;
}

export interface PeopleTracking {
  track_id: string;
  person_id: string;
  person_name: string;
  person_status: string;
  recognition_type: string;
  match_confidence: number;
  detection_confidence: number;
  timestamp: string;
  location: string;
  camera: string;
}

export interface Alert {
  track_id: number | string;
  person_id: string | null;
  person_name: string;
  person_status: string;
  person_label: string;
  alert_type: string;
  recognition_type: string | null;
  match_confidence: number | null;
  detection_confidence: number | null;
  timestamp: number;
  message: string;
}

export interface Status {
  status: string;
  service: string;
  version: string;
  services: {
    ai_service: any;
    kafka_connected: boolean;
    data_stream_running: boolean;
  };
  counts: {
    people_inside: number;
    line_events: number;
    alerts: number;
  };
  crossing_data: {
    In: number;
    Out: number;
    Current_Inside: number;
  };
}

class ApiService {
  private async fetchData<T>(endpoint: string): Promise<T> {
    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error(`Error fetching ${endpoint}:`, error);
      throw error;
    }
  }

  // Video endpoints
  async getVideos(): Promise<{ status: string; count: number; videos: Video[] }> {
    return this.fetchData('/api/v1/videos');
  }

  getVideoUrl(filename: string): string {
    return `${API_BASE_URL}/api/v1/video/${filename}`;
  }

  // Person counting endpoints
  async getPersonCounts(): Promise<PersonCount> {
    return this.fetchData('/api/v1/person-counts');
  }

  async getPeopleTracking(limit: number = 100): Promise<{ status: string; message: string; data: PeopleTracking[]; total_tracks: number }> {
    return this.fetchData(`/api/v1/people-tracking?limit=${limit}`);
  }

  async getPersonRecognition(limit: number = 100): Promise<{ status: string; message: string; data: PeopleTracking[]; statistics: any }> {
    return this.fetchData(`/api/v1/person-recognition?limit=${limit}`);
  }

  // Alert endpoints
  async getAlerts(): Promise<{ alerts: Alert[]; total_alerts: number }> {
    return this.fetchData('/alert');
  }

  // Status endpoints
  async getStatus(): Promise<Status> {
    return this.fetchData('/status');
  }

  async getHealth(): Promise<any> {
    return this.fetchData('/api/v1/health');
  }

  // People inside endpoints
  async getPeopleInside(): Promise<{ total_inside: number }> {
    return this.fetchData('/PeopleInside');
  }

  // People crossing endpoints
  async getPeopleCrossing(): Promise<{ Count: { In: number; Out: number; Current_Inside: number } }> {
    return this.fetchData('/peoplecrossing');
  }

  // Line events endpoints
  async getLineEvents(): Promise<{ events: any[]; total_events: number }> {
    return this.fetchData('/LineEvents');
  }

  // Demo endpoints
  async getDemoLocations(): Promise<{ status: string; locations: any }> {
    return this.fetchData('/api/v1/demo/locations');
  }

  async getDemoStats(): Promise<{ status: string; stats: any }> {
    return this.fetchData('/api/v1/demo/stats');
  }
}

export const apiService = new ApiService();
