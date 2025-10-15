import React from 'react'

const Line = ({className}:{className?:string}) => {
  return (
    <div className={`p-[.2px] ${className} bg-black/10 dark:bg-gray-700 w-full transition-all duration-300`}></div>
  )
}

export default Line