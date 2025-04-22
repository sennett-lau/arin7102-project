"use client"

import Link from "next/link"
import { FC } from "react"

interface BrandProps {
  theme?: "dark" | "light"
}

export const Brand: FC<BrandProps> = ({ theme = "dark" }) => {
  return (
    <Link
      className="flex cursor-pointer flex-col items-center hover:opacity-50"
      href="https://www.chatbotui.com"
      target="_blank"
      rel="noopener noreferrer"
    >
      <div className="mb-2 text-[150px]">
        {/* <ChatbotUISVG theme={theme === "dark" ? "dark" : "light"} scale={0.3} /> */}
        🚽
        {/* <img src="https://files.slack.com/files-pri/T024FNPB6-F07P7S6U2P6/toilet-icon.png" alt="Toilet" className="size-10" width={40} height={40} /> */}
      </div>

      <div className="text-4xl font-bold tracking-wide">Pharmacy Product Shop Assistant</div>
    </Link>
  )
}
