import IconHeader from "../components/headers/IconHeader";
import Line from "../components/items/Line";
import LineChart from "../components/charts/LineChart";
import PieChart from "../components/charts/PieChart";
import DataTable from "../components/items/DataTable";
import { BiCamera, BiRefresh } from "react-icons/bi";
import { RiAlertLine } from "react-icons/ri";
import { AiOutlineLineChart, AiOutlineUsergroupAdd } from "react-icons/ai";
import { FaUsers, FaExclamationTriangle } from "react-icons/fa";
import { MdAnalytics } from "react-icons/md";
import { BsDot } from "react-icons/bs";
import { IoClose } from "react-icons/io5";
import { FiFilter } from "react-icons/fi";
import { useEffect, useState } from "react";
import { FaUserCheck, FaUserTie } from "react-icons/fa";
import { apiService } from "../services/api";
import type { Video, Alert } from "../services/api";
import { usePolling } from "../hooks/usePolling";


const ProtoDashboard = () => {
  const gridStyle = "foreground p-5 rounded-2xl shadow-minimal";

  // Only alerts and status polling - 5 seconds
  const { data: alerts } = usePolling(
    () => apiService.getAlerts(),
    { interval: 10000 }
  );

  const { data: status } = usePolling(
    () => apiService.getStatus(),
    { interval: 10000 }
  );

  // Video call only once - no polling
  const [videos, setVideos] = useState<any>(null);
  const [videosLoading, setVideosLoading] = useState(true);

  useEffect(() => {
    const fetchVideos = async () => {
      try {
        const videoData = await apiService.getVideos();
        setVideos(videoData);
      } catch (error) {
        console.error('Error fetching videos:', error);
      } finally {
        setVideosLoading(false);
      }
    };
    fetchVideos();
  }, []);

  const refreshVideos = async () => {
    setVideosLoading(true);
    try {
      const videoData = await apiService.getVideos();
      setVideos(videoData);
    } catch (error) {
      console.error('Error refreshing videos:', error);
    } finally {
      setVideosLoading(false);
    }
  };

  // Generate data from alerts only
  const getData = () => {
    const defaultLocations = ['Chennai', 'Madurai', 'Trichy', 'Coimbatore', 'Salem'];
    const defaultShops = ['Tambaram', 'Velachery', 'Indiranagar', 'Koramangala', 'Anna Nagar', 'T. Nagar'];
    
    const data: any = {};
    
    // Get people based on name (not containing "person" or "unknown") and prevent duplicates
    const knownPeople = alerts?.alerts?.filter((alert: Alert) => {
      const name = alert.person_name?.toLowerCase() || '';
      return name && 
             !name.includes('person') && 
             !name.includes('unknown') && 
             name.trim() !== '';
    }) || [];
    
    // Remove duplicates based on person_name
    const uniquePeople = knownPeople.reduce((acc: Alert[], current: Alert) => {
      const exists = acc.find(person => person.person_name === current.person_name);
      if (!exists) {
        acc.push(current);
      }
      return acc;
    }, []);
    
    // Get only long_standing and blocked_person alerts for incidents
    const filteredAlerts = alerts?.alerts?.filter((alert: Alert) => 
      alert.alert_type === "long_standing" || alert.alert_type === "blocked_person"
    ) || [];
    
    // Count employees and customers based on person_label from unique people
    const employeeCount = uniquePeople.filter((alert: Alert) => 
      alert.person_label === "employee"
    ).length;
    
    const customerCount = uniquePeople.filter((alert: Alert) => 
      alert.person_label === "customer"
    ).length;
    
    defaultLocations.forEach((location) => {
      data[location.toLowerCase()] = {};
      defaultShops.forEach((shop, shopIndex) => {
        const shopKey = shop.toLowerCase();
        
        // Calculate stats from alerts and status data
        const totalCustomers = customerCount;
        const currentCustomers = status?.counts?.people_inside || uniquePeople.length;
        const totalEmployees = employeeCount;
        const totalIncidents = alerts?.alerts?.length || 0;
        
        data[location.toLowerCase()][shopKey] = {
          cards: [
            { 
              title: "total customers", 
              paragraph: "Total visitors today", 
              value: totalCustomers, 
              icon: FaUsers, 
              iconStyle: "bg-tertiary text-white text-lg p-2 rounded-full !text-sm xxl:text-2xl" 
            },
            { 
              title: "current customers", 
              paragraph: "Customers currently in shop", 
              value: currentCustomers, 
              icon: FaUserCheck, 
              iconStyle: "bg-quinary text-white text-lg p-2 rounded-full !text-sm xxl:text-2xl" 
            },
            { 
              title: "total employees", 
              paragraph: "Employees on duty", 
              value: totalEmployees, 
              icon: FaUserTie, 
              iconStyle: "bg-nonary text-white text-lg p-2 rounded-full !text-sm xxl:text-2xl" 
            },
            { 
              title: "total incidents", 
              paragraph: "Reported incidents today", 
              value: totalIncidents, 
              icon: FaExclamationTriangle, 
              iconStyle: "bg-quaternary text-white text-lg p-2 rounded-full !text-sm xxl:text-2xl" 
            },
          ],
          graph: [
            { name: "9 AM", uv: 15 + shopIndex }, { name: "10 AM", uv: 22 + shopIndex }, { name: "11 AM", uv: 18 + shopIndex },
            { name: "12 PM", uv: 25 + shopIndex }, { name: "1 PM", uv: 30 + shopIndex }, { name: "2 PM", uv: 28 + shopIndex },
            { name: "3 PM", uv: 20 + shopIndex }, { name: "4 PM", uv: 16 + shopIndex }, { name: "5 PM", uv: 12 + shopIndex },
            { name: "6 PM", uv: 18 + shopIndex }, { name: "7 PM", uv: 24 + shopIndex }, { name: "8 PM", uv: 15 + shopIndex },
            { name: "9 PM", uv: 10 + shopIndex },
          ],
          people: uniquePeople.slice(0, 10).map((alert: Alert, index: number) => {
            return {
              id: `#${String(index + 1).padStart(2, '0')}`,
              name: alert.person_name || `Person ${index + 1}`,
              date: new Date(alert.timestamp * 1000).toLocaleDateString(),
              time: new Date(alert.timestamp * 1000).toLocaleTimeString(),
              type: alert.person_label, // Use person_label directly (customer/employee)
              lastSeen: new Date(alert.timestamp * 1000).toLocaleDateString(),
              info: alert.recognition_type || 'Regular visitor',
              incidents: []
            };
          }),
          incidents: filteredAlerts.slice(0, 10).map((alert: Alert, index: number) => ({
            location: location,
            shop: shop,
            date: new Date(alert.timestamp * 1000).toLocaleDateString(),
            time: new Date(alert.timestamp * 1000).toLocaleTimeString(),
            thumbnail: alert.message,
            behaviour: alert.recognition_type || "Unknown",
            alert_type: alert.alert_type,
            confidence: Math.round((alert.match_confidence || 0) * 100),
            camera: `Camera ${index + 1}`,
            person_name: alert.person_name,
            person_status: alert.person_status
          })),
        };
      });
    });
    
    return data;
  };

  const data = getData();



  const locationsList = Object.keys(data);

  const [selectedLocation, setSelectedLocation] = useState(locationsList[0]);
  const [selectedShop, setSelectedShop] = useState(Object.keys(data[locationsList[0]])[0]);
  const [selectedCustomer, setSelectedCustomer] = useState<any>(null);
  const [showCustomerModal, setShowCustomerModal] = useState(false);
  const [selectedIncident, setSelectedIncident] = useState<any>(null);
  const [showIncidentFilterModal, setShowIncidentFilterModal] = useState(false);
  const [showActionModal, setShowActionModal] = useState(false);
  const [selectedVideo, setSelectedVideo] = useState<Video | null>(null);
  const [showVideoModal, setShowVideoModal] = useState(false);

  const currentData = data[selectedLocation][selectedShop];

  const handleLocationChange = (loc: string) => {
    setSelectedLocation(loc);
    setSelectedShop(Object.keys(data[loc])[0]);
    setSelectedIncident(null);
  };
  const handleShopChange = (shop: string) => {
    setSelectedShop(shop);
    setSelectedIncident(null);
  };

  const handleCustomerClick = (person: any) => {
    setSelectedCustomer(person);
    setShowCustomerModal(true);
  };
  

  const handleVideoClick = (video: Video) => {
    setSelectedVideo(video);
    setShowVideoModal(true);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  const iconBgClasses = ["bg-quinary", "bg-secondary", "bg-senary", "bg-quaternary"];

  const AnalyticSection = () => {
  return (
    <section className={`${gridStyle} shadow-minimal`}>
      <div className="space-y-5">
        <div className="flex flex-col xs:flex-row justify-between gap-4 xs:items-center">
          <IconHeader
            headerData={{
              title: "analytics",
              icon: MdAnalytics,
              paragraph: "overview of the analytics",
              iconStyle: "bg-primary text-white",
            }}
          />
          <div className="flex flex-col md:flex-row gap-3 md:items-center flex-wrap">
            <select
              className="background paragraph font-medium py-2 px-3 rounded-lg"
              value={selectedLocation}
              onChange={(e) => handleLocationChange(e.target.value)}
            >
              {locationsList.map((loc, idx) => <option key={idx} value={loc}>{loc}</option>)}
            </select>
            <select
              className="background paragraph font-medium py-2 px-3 rounded-lg"
              value={selectedShop}
              onChange={(e) => handleShopChange(e.target.value)}
            >
              {Object.keys(data[selectedLocation]).map((shopName, idx) => <option key={idx} value={shopName}>{shopName}</option>)}
            </select>
          </div>
        </div>

        <Line />

        <div className="flex flex-col gap-5"> 
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-5"> 
            {currentData.cards.map((card: any, idx: number) => ( 
                <div key={idx} className="background p-4 rounded-lg space-y-3"> 
                  <IconHeader headerData={{ title: card.title, icon: card.icon, iconStyle: `${iconBgClasses[idx]} text-white text-lg p-2 rounded-full !text-sm xxl:text-2xl`, titleStyle: "!text-sm", paragraphStyle: "!text-[10px]", }} /> 
                  <h1 className="text-2xl xxl:text-3xl font-semibold text">{card.value}</h1> 
              </div>
            ))} 
          </div> 
          <div className="flex flex-col xl:flex-row gap-5"> 
            <div className="xl:w-[67.5%] background w-full h-[300px] p-5 rounded-lg"> 
              <LineChart data={currentData.graph} graphColor="#1E90FF" /> 
            </div> 
            <div className="xl:w-[32.5%] background w-full h-[300px] p-5 rounded-lg"> 
                <IconHeader headerData={{ title: "regular vs new", paragraph: "Customer type comparison", icon: AiOutlineLineChart, iconStyle: "text-white bg-septenary", titleStyle: "!text-sm", paragraphStyle: "!text-[11px]", }} /> 
                <Line className="mt-5 mb-2" /> 
                <PieChart data={[{ name: "regular", value: 400, color: "#01B075" }, { name: "new", value: 500, color: "#FF6B6B" }]} graphColor="#01B075" /> 
            </div> 
          </div> 
        </div>
      </div>
    </section>
  );
};


const CustomerSection = () => {
  const [selectedType, setSelectedType] = useState("all");
  const filteredPeople = currentData.people.filter((p: any) =>
    selectedType === "all" ? true : p.type === selectedType
  );

  return (
    <section className={`${gridStyle}`}>
      <div className="space-y-5">
        <div className="flex flex-col xs:flex-row justify-between gap-8 xs:items-center">
          <IconHeader
            headerData={{
              title: "People",
              icon: AiOutlineUsergroupAdd,
              paragraph: "Customer and employee overview",
              iconStyle: "bg-septenary text-white",
            }}
          />
          <div className="flex gap-5 items-center">
            <select
              className="background py-2 px-3 label rounded-lg"
              value={selectedType}
              onChange={(e) => setSelectedType(e.target.value)}
            >
              <option value="all">All</option>
              <option value="customer">Customer</option>
              <option value="employee">Employee</option>
            </select>
          </div>
        </div>
        <Line />
        <div className="max-h-[590px] xxl:max-h-[430px] parent-nav overflow-y-auto grid grid-cols-1 md:grid-cols-2 xxl:grid-cols-3 gap-5">
          {filteredPeople.map((person: any) => (
            <div
              key={person.id}
              className="flex gap-5 background p-3 rounded-lg cursor-pointer"
              onClick={() => handleCustomerClick(person)}
            >
              <div className="w-1/4 foreground flex justify-center items-center rounded-lg">
                <div className="text-secondary text-3xl">
                  <BsDot />
                </div>
              </div>
              <div>
                <h1 className="text font-semibold text-base mb-2 hover:underline">{person.name}</h1>
                <h2 className="paragraph text-xs mb-2">ID - {person.id}</h2>
                <p className="paragraph text-xs mb-2">{person.lastSeen} | {person.time}</p>
                <h2 className="paragraph text-xs mb-2">
                  Type - <span className={
                    person.type === "customer" ? "text-positive-light" : 
                    person.type === "employee" ? "text-blue-500" : 
                    "text-gray-500"
                  }>{person.type}</span>
                </h2>
              </div>
            </div>
          ))}
        </div>
        {showCustomerModal && selectedCustomer && (
          <div className="fixed inset-0 bg-opacity-30 backdrop-blur-sm flex justify-center items-start z-50 overflow-y-auto">
            <div className="foreground border border-color shadow-minimal space-y-8 parent-nav rounded-xl w-[80%] lg:w-[50%] mt-20 p-5">
              <div className="flex justify-between mb-4">
                <IconHeader
                  headerData={{
                    title: selectedCustomer.name,
                    icon: AiOutlineUsergroupAdd,
                    paragraph: `${selectedCustomer.type.charAt(0).toUpperCase() + selectedCustomer.type.slice(1)} details overview`,
                    iconStyle: "bg-secondary text-white",
                  }}
                />
                <button className="text text-2xl" onClick={() => setShowCustomerModal(false)}>
                  <IoClose />
                </button>
              </div>
              <Line />
              <div className="space-y-4">
                <p className="text text-sm"><span className="paragraph">ID:</span> {selectedCustomer.id}</p>
                <p className="text text-sm"><span className="paragraph">Type:</span> <span className="text-positive-light">{selectedCustomer.type}</span></p>
                <p className="text text-sm"><span className="paragraph">Last Seen:</span> {selectedCustomer.lastSeen}</p>
                <p className="text text-sm"><span className="paragraph">Info:</span> {selectedCustomer.info}</p>
                <p className="text text-sm"><span className="paragraph">Linked Incidents:</span> {selectedCustomer.incidents.length}</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
};


  const IncidentSection = () => {
    
  const incidentColumns = [
    { 
      label: "Message", 
      render: (incident: any) => (
        <div className="max-w-xs lg:max-w-md xl:max-w-lg">
          <span className="text-xs break-words">{incident.thumbnail}</span>
        </div>
      )
    },
    { 
      label: "Type", 
      render: (incident: any) => {
        // Map alert_type to display text
        let typeText = "Unknown";
        let typeClass = "bg-gray-800 text-white";
        
        if (incident.alert_type === "long_standing") {
          typeText = "Long Standing";
          typeClass = "bg-blue-900 text-white";
        } else if (incident.alert_type === "blocked_person") {
          typeText = "Blocked Person";
          typeClass = "bg-red-900 text-white";
        }
        
        return (
          <span className={`px-3 py-1 rounded-md text-xs font-medium ${typeClass}`}>
            {typeText}
          </span>
        );
      }
    },
    { label: "Date & Time", render: (incident: any) => `${incident.date} ${incident.time}` },
  ];

    return (
      <section className="foreground p-5 rounded-2xl shadow-minimal">
        <div className="space-y-5">
          <div className="flex flex-col xs:flex-row justify-between gap-8 xs:items-center">
            <IconHeader
              headerData={{
                title: "Incidents",
                icon: RiAlertLine,
                paragraph: "Incident overview summary",
                iconStyle: "bg-quaternary text-white",
              }}
            />
            <div className="flex items-center gap-4">
              <button
                className="text-white bg-primary rounded-lg py-2 px-4 text-sm capitalize font-semibold flex items-center gap-2"
                onClick={() => setShowIncidentFilterModal(true)}
              >
                <FiFilter /> Filter
              </button>
            </div>
          </div>
          <Line />
          <div className="">
            <DataTable data={currentData.incidents} columns={incidentColumns} minRows={6} pagination={true}/>
          </div>
        </div>

        {showIncidentFilterModal && (
          <div className="fixed inset-0 bg-black/30 backdrop-blur-sm flex justify-center items-start z-50 overflow-y-auto">
            <div className="foreground border border-color shadow-minimal space-y-8 parent-nav rounded-xl w-[80%] lg:w-[40%] mt-20 p-5">
              <div className="flex justify-between mb-4">
                <IconHeader
                  headerData={{
                    title: 'Filter Incidents',
                    icon: AiOutlineUsergroupAdd,
                    paragraph: "Filter customer incidents",
                    iconStyle: "bg-secondary text-white",
                  }}
                />
                <button className="text text-2xl" onClick={() => setShowIncidentFilterModal(false)}>
                  <IoClose />
                </button>
              </div>
              <Line />
              <div className="space-y-4">
                <div><label className="label">Location</label><select className="w-full py-2 px-3 text rounded-lg background mt-2" value={selectedLocation} onChange={e => handleLocationChange(e.target.value)}>
                  {locationsList.map((loc, idx) => <option key={idx} value={loc}>{loc}</option>)}
                </select></div>
                <div><label className="label">Shop</label><select className="w-full text py-2 px-3 rounded-lg background mt-2" value={selectedShop} onChange={e => handleShopChange(e.target.value)}>
                  {Object.keys(data[selectedLocation]).map((shop, idx) => <option key={idx} value={shop}>{shop}</option>)}
                </select></div>
                <div className="flex justify-end gap-3">
                  <button className="text-white text font-title bg-primary rounded-lg px-4 py-1">Apply</button>
                  <button className="text-white text font-title bg-octonary rounded-lg px-4 py-1">Reset</button>
                </div>
              </div>
            </div>
          </div>
        )}
      </section>
    );
  };

  const PlayerSection = () => (
  <section className={`${gridStyle}`}>
    <div className="space-y-5">
      <div className="flex flex-col md:flex-row justify-between gap-8 xs:items-center">
        <IconHeader
          headerData={{
            title: "CCTV Footages",
            icon: BiCamera,
            paragraph: "View and analyze footage",
            iconStyle: "bg-quinary text-white",
          }}
        />
        <div className="flex gap-2">
          <button 
            onClick={refreshVideos} 
            className="text-white w-fit bg-secondary rounded-lg py-2 px-4 text-sm capitalize font-semibold flex items-center gap-2"
          >
            <BiRefresh /> Refresh Videos
          </button>
          {selectedIncident && (
            <button onClick={()=> setShowActionModal(true)} className="text-white w-fit bg-primary rounded-lg py-2 px-4 text-sm capitalize font-semibold flex items-center gap-2">
              <FiFilter /> action
            </button>
          )}
        </div>
      </div>
      <Line />

      {/* Video List - Only this section */}
      <div className="space-y-2">
        <h3 className="text-sm font-semibold text">Available Videos ({videos?.count || 0})</h3>
        {videosLoading ? (
          <div className="text-center py-4">
            <span className="text-sm paragraph">Loading videos...</span>
          </div>
        ) : videos?.videos && videos.videos.length > 0 ? (
          <div className="grid grid-cols-1 gap-2">
            {videos.videos.map((video: Video, index: number) => (
              <div
                key={index}
                className="background p-3 rounded-lg cursor-pointer hover:bg-opacity-80 transition-colors"
                onClick={() => handleVideoClick(video)}
              >
                <div className="flex justify-between items-center">
                  <div className="flex-1">
                    <h4 className="text-sm font-medium text truncate">{video.filename}</h4>
                    <p className="text-xs paragraph">
                      {formatDate(video.created_time)} • {formatFileSize(video.size)}
                    </p>
                    <p className="text-xs paragraph">
                      {video.location} • {video.camera}
                    </p>
                  </div>
                  <div className="text-quinary text-lg">
                    <BiCamera />
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-4">
            <span className="text-sm paragraph">No videos available</span>
          </div>
        )}
      </div>
    </div>

    {/* Video Modal */}
    {showVideoModal && selectedVideo && (
      <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex justify-center items-center z-50">
        <div className="foreground border border-color shadow-minimal rounded-xl w-[90%] lg:w-[70%] max-w-4xl p-5">
          <div className="flex justify-between mb-4">
            <IconHeader
              headerData={{
                title: selectedVideo.filename,
                icon: BiCamera,
                paragraph: "Video playback and details",
                iconStyle: "bg-quinary text-white",
              }}
            />
            <button className="text text-2xl" onClick={() => setShowVideoModal(false)}>
              <IoClose />
            </button>
          </div>
          <Line />
          <div className="space-y-4">
            <video
              controls
              className="w-full h-[400px] bg-black rounded-lg"
              src={apiService.getVideoUrl(selectedVideo.filename)}
            >
              Your browser does not support the video tag.
            </video>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="paragraph">Location:</span>
                <p className="text">{selectedVideo.location}</p>
              </div>
              <div>
                <span className="paragraph">Camera:</span>
                <p className="text">{selectedVideo.camera}</p>
              </div>
              <div>
                <span className="paragraph">Size:</span>
                <p className="text">{formatFileSize(selectedVideo.size)}</p>
              </div>
              <div>
                <span className="paragraph">Created:</span>
                <p className="text">{formatDate(selectedVideo.created_time)}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    )}

    {/* Action Modal */}
    {showActionModal && (
      <div className="fixed inset-0 bg-opacity-30 backdrop-blur-sm flex justify-center items-start z-50 overflow-y-auto">
        <div className="foreground border border-color shadow-minimal space-y-8 parent-nav rounded-xl w-[80%] lg:w-[20%] mt-20 p-5">
          <div className="flex justify-between mb-4">
            <IconHeader
              headerData={{
                title: 'select action',
                icon: AiOutlineUsergroupAdd,
                paragraph: `details overview`,
                iconStyle: "bg-secondary text-white",
              }}
            />
            <button className="text text-2xl" onClick={() => setShowActionModal(false)}>
              <IoClose />
            </button>
          </div>
          <Line />
          <div className="flex flex-col gap-3 lg:flex-row">
            <button className="bg-senary px-4 py-1 text font-medium flex-1 rounded-lg">snooze</button>
            <button className="bg-septenary px-4 py-1 text font-medium flex-1 rounded-lg">escalate</button>
            <button className="bg-quaternary px-4 py-1 text font-medium flex-1 rounded-lg">close</button>
          </div>
        </div>
      </div>
    )}
  </section>
);


  return (
    <main>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-2">
        <AnalyticSection />
        <CustomerSection />
        <IncidentSection />
        <PlayerSection />
      </div>
    </main>
  );
};

export default ProtoDashboard;
