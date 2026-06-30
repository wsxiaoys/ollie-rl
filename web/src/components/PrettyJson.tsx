import { HTMLAttributes } from "react";

interface PrettyJsonProps extends Omit<HTMLAttributes<HTMLElement>, "children"> {
  data: unknown;
  expand?: number | string;
  truncateString?: number | string;
}

export function PrettyJson({
  data,
  expand = 1,
  truncateString,
  className,
  ...props
}: PrettyJsonProps) {
  const jsonString = typeof data === "string" ? data : JSON.stringify(data);

  // Use a dynamic tag to completely bypass TypeScript JSX intrinsic element checks
  const Tag = "pretty-json" as any;

  return (
    <Tag
      expand={expand?.toString()}
      truncate-string={truncateString?.toString()}
      class={className}
      {...props}
    >
      {jsonString}
    </Tag>
  );
}

export default PrettyJson;
