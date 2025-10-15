import { useEffect, useState } from "react";
import { BiBell, BiMoon, BiSearch, BiSun } from "react-icons/bi";
import IconHeader from "../headers/IconHeader";
import Line from "../items/Line";
import { IoClose } from "react-icons/io5";
import { HiMiniBars3BottomRight } from "react-icons/hi2";

const Header = () => {
  const [isScrolled, setIsScrolled] = useState(false);
  const [darkMode, setDarkMode] = useState(true);
  const [showMessage, setShowMessage] = useState(false);
  const [nav, setNav] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 0);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <header
      className={`sticky top-0 z-40 transition-all duration-300 pt-3 pb-2 background ${
        isScrolled ? "" : ""
      }`}
    >
      <div className="main foreground p-5 rounded-2xl flex justify-between items-center shadow-minimal relative">
        {/* Logo */}
        <div className="logo flex gap-5 items-center">
          <div className="lg:h-[50px] w-[20%] md:w-[10%] xl:w-[8%] xxl:w-[6%]">
            <img src="/logo.png" className="object-cover w-full" alt="" />
          </div>
          <div>
            <h1 className="text font-semibold text-xl md:text-2xl flex gap-2">
              <span>
                paarv<span className="text-nonary">AI</span>
              </span>
              <span className="md:block hidden">dashboard</span>
            </h1>
            <h2 className="text paragraph text-xs md:text-sm font-paragraph">
              the intelligent eye
            </h2>
          </div>
        </div>
        <button
          className="block lg:hidden text text-3xl"
          onClick={() => setNav(!nav)}
        >
          {nav ? <IoClose /> : <HiMiniBars3BottomRight />}
        </button>
        
        <div className="hidden lg:flex gap-5 items-center">
          <div className="border-r border-gray-800 flex gap-5 pr-5">
            <button className="text background p-3 text-lg rounded-full">
              <BiSearch />
            </button>
            <button
              className="text background p-3 text-lg rounded-full"
              onClick={() => {
                setShowMessage(true);
              }}
            >
              <BiBell />
            </button>
            <button
              className="text background p-3 text-lg rounded-full"
              onClick={() => setDarkMode(!darkMode)}
            >
              {darkMode ? <BiMoon /> : <BiSun />}
            </button>
          </div>
          <div className="flex items-center gap-3">
            <div className="rounded-full px-5 py-5 background"></div>
            <div>
              <h1 className="capitalize text">admin</h1>
              <h4 className="paragraph text-xs">role</h4>
            </div>
          </div>
        </div>
        {showMessage && (
          <div className="fixed inset-0 bg-opacity-30 backdrop-blur-sm flex justify-center items-start z-50 overflow-y-auto">
            <div className="foreground shadow-minimal border border-color space-y-8 rounded-xl w-[80%] lg:w-[30%] mt-20 p-5">
              <div className="flex justify-between items-center mb-4">
                <IconHeader
                  headerData={{
                    title: "notification",
                    icon: BiBell,
                    paragraph: "overview of the analytics",
                    iconStyle: "bg-septenary text-white",
                  }}
                />
                <button
                  className="text text-2xl"
                  onClick={() => setShowMessage(false)}
                >
                  <IoClose />
                </button>
              </div>
              <Line />
              <div className="space-y-4 h-[400px] xl:h-[400px] xxl:h-[550px] overflow-y-auto parent-nav"></div>
            </div>
          </div>
        )}
      </div>
      <div
          className={`absolute backdrop-blur-sm top-full h-screen left-0 w-full bg-foreground border-t border-color rounded-b-2xl transform transition-transform duration-300 lg:hidden ${
            nav ? "translate-x-0" : "-translate-x-full"
          }`}
        >
          <div className="flex main flex-col p-5 gap-4 h-[82%] foreground shadow-minimal rounded-2xl mt-5">
            <button
              className="flex items-center gap-3 text"
              onClick={() => {
                setShowMessage(true);
                setNav(false);
              }}
            >
              <BiBell className="text-lg" /> Notifications
            </button>
            <button
              className="flex items-center gap-3 text"
              onClick={() => setDarkMode(!darkMode)}
            >
              {darkMode ? (
                <>
                  <BiMoon className="text-lg" /> Dark Mode
                </>
              ) : (
                <>
                  <BiSun className="text-lg" /> Light Mode
                </>
              )}
            </button>
            <button className="flex items-center gap-3 text">
              <BiSearch className="text-lg" /> Search
            </button>
            <div className="flex items-center gap-3 background p-3 rounded-2xl">
              <div className="rounded-full px-5 py-5 foreground"></div>
              <div>
                <h1 className="capitalize text">admin</h1>
                <h4 className="paragraph text-xs">role</h4>
              </div>
            </div>
          </div>
        </div>
    </header>
  );
};

export default Header;
