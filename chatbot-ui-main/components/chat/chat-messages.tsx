import { useChatHandler } from "@/components/chat/chat-hooks/use-chat-handler"
import { ChatbotUIContext } from "@/context/context"
import { Tables } from "@/supabase/types"
import { FC, useContext, useEffect, useState } from "react"
import { Message } from "../messages/message"

interface ChatMessagesProps {}

export const ChatMessages: FC<ChatMessagesProps> = ({}) => {
  const { chatMessages, chatFileItems } = useContext(ChatbotUIContext)

  const { handleSendEdit } = useChatHandler()

  const [editingMessage, setEditingMessage] = useState<Tables<"messages">>()

  const sortedChatMessages = chatMessages
    .sort((a, b) => a.message.sequence_number - b.message.sequence_number)
    // .map((chatMessage, index, array) => {
    //   const messageFileItems = chatFileItems.filter(
    //     (chatFileItem, _, self) =>
    //       chatMessage.fileItems.includes(chatFileItem.id) &&
    //       self.findIndex(item => item.id === chatFileItem.id) === _
    //   )

    //   return (
    //     <Message
    //       key={chatMessage.message.sequence_number}
    //       message={chatMessage.message}
    //       fileItems={messageFileItems}
    //       isEditing={editingMessage?.id === chatMessage.message.id}
    //       isLast={index === array.length - 1}
    //       onStartEdit={setEditingMessage}
    //       onCancelEdit={() => setEditingMessage(undefined)}
    //       onSubmitEdit={handleSendEdit}
    //     />
    //   )
    // })
    const userMessage = chatMessages.at(-2)!
    const botMessage = chatMessages.at(-1)!
    
    return (
    <div className="flex flex-col">
        <Message
           key={userMessage.message.sequence_number}
           message={userMessage.message}
           fileItems={[]}
           isEditing={editingMessage?.id === userMessage.message.id}
           isLast={false}
           onStartEdit={setEditingMessage}
           onCancelEdit={() => setEditingMessage(undefined)}
           onSubmitEdit={handleSendEdit}
         />
         <div className="flex justify-center">
        </div>
        <Message
           key={botMessage.message.sequence_number}
           message={botMessage.message}
           fileItems={[]}
           isEditing={editingMessage?.id === botMessage.message.id}
           isLast={false}
           onStartEdit={setEditingMessage}
           onCancelEdit={() => setEditingMessage(undefined)}
           onSubmitEdit={handleSendEdit}
         />
    </div>
    )
}
