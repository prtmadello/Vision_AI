import IconHeader from "../components/headers/IconHeader";
import Line from "../components/items/Line";
import LineChart from "../components/charts/LineChart";
import PieChart from "../components/charts/PieChart";
import DataTable from "../components/items/DataTable";
import { BiCamera, BiLeftArrowAlt, BiPause, BiPlay, BiRefresh, BiRightArrowAlt } from "react-icons/bi";
import { RiAlertLine } from "react-icons/ri";
import { AiOutlineLineChart, AiOutlineUsergroupAdd } from "react-icons/ai";
import { FaUsers, FaExclamationTriangle } from "react-icons/fa";
import { MdAnalytics } from "react-icons/md";
import { BsDot } from "react-icons/bs";
import { IoClose } from "react-icons/io5";
import { FiFilter } from "react-icons/fi";
import { useEffect, useRef, useState } from "react";
import { Swiper, SwiperSlide } from 'swiper/react';
import 'swiper/css';
import 'swiper/css/navigation';
import 'swiper/css/pagination';

import { FaUserCheck, FaUserTie, FaExclamationCircle } from "react-icons/fa";


const Dashboard = () => {
  const gridStyle = "foreground p-5 rounded-2xl shadow-minimal";

const data: any = {
  chennai: {
    tambaram: {
      cards: [
        { title: "total customers", paragraph: "Total visitors today", value: 1200, icon: FaUsers, iconStyle: "bg-tertiary text-white text-lg p-2 rounded-full !text-sm xxl:text-2xl" },
        { title: "current customers", paragraph: "Customers currently in shop", value: 5, icon: FaUserCheck, iconStyle: "bg-quinary text-white text-lg p-2 rounded-full !text-sm xxl:text-2xl" },
        { title: "total employees", paragraph: "Employees on duty", value: 5, icon: FaUserTie, iconStyle: "bg-nonary text-white text-lg p-2 rounded-full !text-sm xxl:text-2xl" },
        { title: "total incidents", paragraph: "Reported incidents today", value: 5, icon: FaExclamationTriangle, iconStyle: "bg-quaternary text-white text-lg p-2 rounded-full !text-sm xxl:text-2xl" },
      ],
      graph: [
        { name: "9 AM", uv: 15 }, { name: "10 AM", uv: 22 }, { name: "11 AM", uv: 18 },
        { name: "12 PM", uv: 25 }, { name: "1 PM", uv: 30 }, { name: "2 PM", uv: 28 },
        { name: "3 PM", uv: 20 }, { name: "4 PM", uv: 16 }, { name: "5 PM", uv: 12 },
        { name: "6 PM", uv: 18 }, { name: "7 PM", uv: 24 }, { name: "8 PM", uv: 15 },
        { name: "9 PM", uv: 10 },
      ],
      people: [
        {
          id: "#01", name: "John Doe", date: "10.10.2025", time: "11:00 AM", type: "customer", lastSeen: "09.10.2025", info: "Frequent visitor",
          incidents: [
            { location: "Tambaram", date: "10.10.2025", type: "Suspicious", confidence: 90, camera: "Entrance Cam 1" },
            { location: "Tambaram", date: "11.10.2025", type: "Stays Here", confidence: 85, camera: "Cash Counter Cam" },
          ],
        },
        { id: "#02", name: "Ava Smith", date: "10.10.2025", time: "10:45 AM", type: "customer", lastSeen: "08.10.2025", info: "Regular customer", incidents: [] },
        { id: "#03", name: "Michael Johnson", date: "11.10.2025", time: "12:30 PM", type: "employee", lastSeen: "11.10.2025", info: "Store manager", incidents: [] },
      ],
      incidents: [
        { location: "Chennai", shop: "Tambaram", date: "10.10.2025", thumbnail: "dummy", behaviour: "Suspicious", confidence: 90, camera: "Entrance Cam 1" },
        { location: "Chennai", shop: "Tambaram", date: "10.10.2025", thumbnail: "dummy", behaviour: "Stays Here", confidence: 90, camera: "Entrance Cam 2" },
      ],
    },
    velachery: {
      cards: [
        { title: "total customers", paragraph: "Total visitors today", value: 900, icon: FaUsers, iconStyle: "bg-secondary text-white text-lg p-2 rounded-full" },
        { title: "current customers", paragraph: "Customers currently in shop", value: 5, icon: FaUserCheck, iconStyle: "bg-quinary text-white text-lg p-2 rounded-full !text-sm xxl:text-2xl" },
        { title: "total employees", paragraph: "Employees on duty", value: 5, icon: FaUserTie, iconStyle: "bg-nonary text-white text-lg p-2 rounded-full !text-sm xxl:text-2xl" },
        { title: "Total Incidents", paragraph: "Reported incidents today", value: 3, icon: FaExclamationCircle, iconStyle: "bg-quaternary text-white text-lg p-2 rounded-full" },
      ],
      graph: [
        { name: "9 AM", uv: 10 }, { name: "10 AM", uv: 18 }, { name: "11 AM", uv: 15 },
      ],
      people: [
        { id: "#04", name: "Sara Khan", date: "11.10.2025", time: "10:00 AM", type: "customer", lastSeen: "10.10.2025", info: "Regular customer", incidents: [] },
        { id: "#05", name: "Rahul Verma", date: "11.10.2025", time: "11:15 AM", type: "employee", lastSeen: "11.10.2025", info: "Cashier", incidents: [] },
      ],
      incidents: [
        { location: "Chennai", shop: "Velachery", date: "11.10.2025", thumbnail: "dummy", behaviour: "Loitering", confidence: 88, camera: "Entrance Cam 1" },
      ],
    },
  },
  bangalore: {
    indiranagar: {
      cards: [
        { title: "total customers", paragraph: "Total visitors today", value: 1500, icon: FaUsers, iconStyle: "bg-secondary text-white text-lg p-2 rounded-full" },
        { title: "current customers", paragraph: "Customers currently in shop", value: 5, icon: FaUserCheck, iconStyle: "bg-quinary text-white text-lg p-2 rounded-full !text-sm xxl:text-2xl" },
        { title: "total employees", paragraph: "Employees on duty", value: 5, icon: FaUserTie, iconStyle: "bg-nonary text-white text-lg p-2 rounded-full !text-sm xxl:text-2xl" },
        { title: "Total Incidents", paragraph: "Reported incidents today", value: 6, icon: FaExclamationCircle, iconStyle: "bg-quaternary text-white text-lg p-2 rounded-full" },
      ],
      graph: [
        { name: "9 AM", uv: 20 }, { name: "10 AM", uv: 25 }, { name: "11 AM", uv: 22 },
      ],
      people: [
        { id: "#06", name: "Ravi Kumar", date: "10.10.2025", time: "09:30 AM", type: "customer", lastSeen: "09.10.2025", info: "Regular visitor", incidents: [] },
        { id: "#07", name: "Priya Singh", date: "10.10.2025", time: "10:15 AM", type: "employee", lastSeen: "10.10.2025", info: "Security", incidents: [] },
      ],
      incidents: [
        { location: "Bangalore", shop: "Indiranagar", date: "10.10.2025", thumbnail: "dummy", behaviour: "Suspicious", confidence: 90, camera: "Entrance Cam 1" },
        { location: "Bangalore", shop: "Indiranagar", date: "10.10.2025", thumbnail: "dummy", behaviour: "Stays Here", confidence: 82, camera: "Exit Cam" },
      ],
    },
    koramangala: {
      cards: [
        { title: "total customers", paragraph: "Total visitors today", value: 1300, icon: FaUsers, iconStyle: "bg-tertiary text-white text-lg p-2 rounded-full" },
        { title: "current customers", paragraph: "Customers currently in shop", value: 4, icon: FaUserCheck, iconStyle: "bg-quinary text-white text-lg p-2 rounded-full" },
        { title: "total employees", paragraph: "Employees on duty", value: 6, icon: FaUserTie, iconStyle: "bg-nonary text-white text-lg p-2 rounded-full" },
        { title: "Total Incidents", paragraph: "Reported incidents today", value: 2, icon: FaExclamationCircle, iconStyle: "bg-quaternary text-white text-lg p-2 rounded-full" },
      ],
      graph: [
        { name: "9 AM", uv: 18 }, { name: "10 AM", uv: 22 }, { name: "11 AM", uv: 20 },
      ],
      people: [
        { id: "#08", name: "Ananya Rao", date: "11.10.2025", time: "09:45 AM", type: "customer", lastSeen: "10.10.2025", info: "New visitor", incidents: [] },
        { id: "#09", name: "Karan Mehta", date: "11.10.2025", time: "10:30 AM", type: "employee", lastSeen: "11.10.2025", info: "Supervisor", incidents: [] },
      ],
      incidents: [
        { location: "Bangalore", shop: "Koramangala", date: "11.10.2025", thumbnail: "dummy", behaviour: "Loitering", confidence: 92, camera: "Entrance Cam 1" },
      ],
    },
  },
};



  const locationsList = Object.keys(data);

  const [selectedLocation, setSelectedLocation] = useState(locationsList[0]);
  const [selectedShop, setSelectedShop] = useState(Object.keys(data[locationsList[0]])[0]);
  const [selectedCustomer, setSelectedCustomer] = useState<any>(null);
  const [showCustomerModal, setShowCustomerModal] = useState(false);
  const [selectedIncident, setSelectedIncident] = useState<any>(null);
  const [showIncidentFilterModal, setShowIncidentFilterModal] = useState(false);
  const [showActionModal, setShowActionModal] = useState(false)

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
  
  const handleIncidentClick = (incident: any) => {
    console.log('got incedent data in function',incident)
    setSelectedIncident(incident);
    console.log('after setting state',selectedIncident)
  };

  const iconBgClasses = ["bg-quinary", "bg-secondary", "bg-senary", "bg-quaternary"];

  const AnalyticSection = () => {
  const swiperRef = useRef<any>(null);

  const allShopsFlattened = locationsList.flatMap(loc =>
    Object.keys(data[loc]).map(shopName => ({ ...data[loc][shopName], location: loc, shopName }))
  );

  const [currentIndex, setCurrentIndex] = useState(
    allShopsFlattened.findIndex(s => s.location === selectedLocation && s.shopName === selectedShop)
  );

  const [paused, setPaused] = useState(false); 

  useEffect(() => {
    setCurrentIndex(allShopsFlattened.findIndex(s => s.location === selectedLocation && s.shopName === selectedShop));
  }, [selectedLocation, selectedShop]);

  const handlePrev = () => setCurrentIndex((currentIndex - 1 + allShopsFlattened.length) % allShopsFlattened.length);
  const handleNext = () => setCurrentIndex((currentIndex + 1) % allShopsFlattened.length);
  const handleRefresh = () => setCurrentIndex(0);
  const handlePauseToggle = () => setPaused(prev => !prev); 

  useEffect(() => {
    swiperRef.current?.slideTo(currentIndex);
  }, [currentIndex]);

  useEffect(() => {
    const interval = setInterval(() => {
      if (!paused && !selectedIncident) {
        handleNext();
      }
    }, 3000);
    return () => clearInterval(interval);
  }, [currentIndex, paused]);

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

            <div className="flex gap-3 mt-4 md:mt-0">
              <button
                className="text-white bg-primary p-2 text-lg rounded-lg"
                onClick={handlePrev}
              >
                <BiLeftArrowAlt />
              </button>
              <button
                className="text-white bg-primary p-2 text-lg rounded-lg"
                onClick={handlePauseToggle}
              >
                {paused ? <BiPlay /> : <BiPause />}
              </button>

              <button
                className="text-white bg-primary p-2 text-lg rounded-lg"
                onClick={handleRefresh}
              >
                <BiRefresh />
              </button>
              <button
                className="text-white bg-primary p-2 text-lg rounded-lg"
                onClick={handleNext}
              >
                <BiRightArrowAlt />
              </button>
            </div>
          </div>
        </div>

        <Line />

        <Swiper
          onSwiper={(swiper) => (swiperRef.current = swiper)}
          slidesPerView={1}
          spaceBetween={30}
          onSlideChange={(swiper) => {
            const currentShop = allShopsFlattened[swiper.activeIndex];
            setSelectedLocation(currentShop.location);
            setSelectedShop(currentShop.shopName);
            setSelectedIncident(null);
          }}
        >
          {allShopsFlattened.map((shop) => (
            <SwiperSlide key={`${shop.location}-${shop.shopName}`}>
              <div className="flex flex-col gap-5"> 
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-5"> 
                  {shop.cards.map((card: any, idx: number) => ( 
                      <div key={idx} className="background p-4 rounded-lg space-y-3"> 
                        <IconHeader headerData={{ title: card.title, icon: card.icon, iconStyle: `${iconBgClasses[idx]} text-white text-lg p-2 rounded-full !text-sm xxl:text-2xl`, titleStyle: "!text-sm", paragraphStyle: "!text-[10px]", }} /> 
                        <h1 className="text-2xl xxl:text-3xl font-semibold text">{card.value}</h1> 
                    </div>
                  ))} 
                </div> 
                <div className="flex flex-col xl:flex-row gap-5"> 
                  <div className="xl:w-[67.5%] background w-full h-[300px] p-5 rounded-lg"> 
                    <LineChart data={shop.graph} graphColor="#1E90FF" /> 
                  </div> 
                  <div className="xl:w-[32.5%] background w-full h-[300px] p-5 rounded-lg"> 
                      <IconHeader headerData={{ title: "regular vs new", paragraph: "Customer type comparison", icon: AiOutlineLineChart, iconStyle: "text-white bg-septenary", titleStyle: "!text-sm", paragraphStyle: "!text-[11px]", }} /> 
                      <Line className="mt-5 mb-2" /> 
                      <PieChart data={[{ name: "regular", value: 400, color: "#01B075" }, { name: "new", value: 500, color: "#FF6B6B" }]} graphColor="#01B075" /> 
                  </div> 
                </div> 
              </div>
            </SwiperSlide>
          ))}
        </Swiper>
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
                  Type - <span className={person.type === "customer" ? "text-positive-light" : "text-red-500"}>{person.type}</span>
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
      label: "Camera", 
      render: (incident: any) => (
        <span 
          className=" cursor-pointer text-primary rounded-md text-xs font-medium"
          onClick={() => {console.log('from btn :', incident);handleIncidentClick(incident)}}
        >
          {selectedLocation}
        </span>
      )
    },
    { label: "Shop", render: () => selectedShop },
    { label: "Date", render: (incident: any) => incident.date },
    // { 
    //   label: "Camera", 
    //   render: (incident: any) => (
    //     <span 
    //       className=" cursor-pointer text-primary rounded-md text-xs font-medium"
    //       onClick={() => {console.log('from btn :', incident);handleIncidentClick(incident)}}
    //     >
    //       {incident.camera}
    //     </span>
    //   )
    // },
    { label: "Message", render: (incident: any) => incident.thumbnail },
    { 
      label: "behaviour", 
      render: (incident: any) => (
        <span className={`px-3 py-1 rounded-md text-xs font-medium ${incident.type === "Known" ? "bg-green-500/20 text-green-700" : "bg-red-500/20 text-red-700"}`}>
          {incident.behaviour}
        </span>
      ) 
    },
    { label: "Confidence", render: (incident: any) => `${incident.confidence}%` },
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
  <section className={`${gridStyle} flex flex-col h-full`}>
    <div className="space-y-5 flex flex-col flex-grow">
      <div className="flex flex-col md:flex-row justify-between gap-8 xs:items-center">
        <IconHeader
          headerData={{
            title: "CCTV Footages",
            icon: BiCamera,
            paragraph: "View and analyze footage",
            iconStyle: "bg-quinary text-white",
          }}
        />
        {selectedIncident && (
          <button onClick={()=> setShowActionModal(true)} className="text-white w-fit bg-primary rounded-lg py-2 px-4 text-sm capitalize font-semibold flex items-center gap-2">
            <FiFilter /> action
          </button>)}
      </div>
      <Line />

      <div className="flex-grow relative rounded-lg overflow-hidden background w-full h-[400px] md:h-full flex justify-center items-center text-center">
        {selectedIncident ? (
          <>
            <div className="absolute inset-0 bg-black/40 flex justify-center items-center  flex-col">
            <span  className=' cursor-pointer text-white foreground p-2 rounded-lg' onClick={()=>handleIncidentClick(null)}><IoClose/></span>
              <span className="text-white text-sm font-medium">
                {selectedIncident.camera} - {selectedIncident.type}
              </span>
            </div>
            <p className="absolute bottom-2 left-1/2 -translate-x-1/2 text-white text-xs">
              {selectedIncident.date}
            </p>
          </>
        ) : (
          <>
            <div className="flex flex-col justify-center items-center h-full text-white">
              <div className="text-quinary text-4xl mb-2"><BsDot /></div>
              <span className="text-sm font-medium">No incidents selected</span>
              <p className="text-xs mt-1">CCTV feed will appear here</p>
            </div>
          </>
        )}
      </div>
    </div>
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
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <AnalyticSection />
        <CustomerSection />
        <IncidentSection />
        <PlayerSection />
      </div>
    </main>
  );
};

export default Dashboard;
