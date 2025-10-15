import { BiCopyright } from "react-icons/bi";
import { FaFacebookF, FaTwitter, FaLinkedinIn, FaInstagram } from "react-icons/fa";

const Footer = () => {
  return (
    <footer className="foreground py-6 shadow-minimal">
      <div className="main flex flex-col lg:flex-row items-center justify-between gap-4">
        
        <div className="flex gap-4">
          <a href="https://facebook.com" target="_blank" rel="noopener noreferrer" className="hover:text-primary transition !text-xs">
            <FaFacebookF size={18} />
          </a>
          <a href="https://twitter.com" target="_blank" rel="noopener noreferrer" className="hover:text-primary transition !text-xs">
            <FaTwitter size={18} />
          </a>
          <a href="https://linkedin.com" target="_blank" rel="noopener noreferrer" className="hover:text-primary transition !text-xs">
            <FaLinkedinIn size={18} />
          </a>
          <a href="https://instagram.com" target="_blank" rel="noopener noreferrer" className="hover:text-primary transition !text-xs">
            <FaInstagram size={18} />
          </a>
        </div>
        <p className="paragraph leading-0 text-center text-xs lg:text-sm flex items-center gap-1">
            <BiCopyright/> 2025 Paarvai@madello Consulting. All rights reserved.
        </p>
      </div>
    </footer>
  )
}

export default Footer;
